import torch
import torch.nn as nn


class CRF(nn.Module):
    def __init__(self, num_tags, batch_first=True):
        super(CRF, self).__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def forward(self, emissions, tags, mask=None, reduction='sum'):
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.uint8, device=emissions.device)

        mask = mask.bool()

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        seq_length, batch_size = tags.shape

        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # Find the index of the last valid token for each batch
        seq_ends = mask.long().sum(dim=0) - 1
        last_tags = tags.gather(0, seq_ends.unsqueeze(0)).squeeze(0)
        score += self.end_transitions[last_tags]

        log_Z = self._compute_log_partition(emissions, mask)
        ll = score - log_Z

        if reduction == 'none':
            return ll
        elif reduction == 'sum':
            return ll.sum()
        elif reduction == 'mean':
            return ll.mean()
        else:
            return ll.sum()

    def _compute_log_partition(self, emissions, mask):
        seq_length, batch_size, n_tags = emissions.shape
        alpha = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            emission_score = emissions[i].unsqueeze(1).expand(batch_size, n_tags, n_tags)
            trans_score = self.transitions.unsqueeze(0).expand(batch_size, n_tags, n_tags)
            alpha_broadcast = alpha.unsqueeze(2).expand(batch_size, n_tags, n_tags)

            alpha_next = (alpha_broadcast + trans_score + emission_score).logsumexp(dim=1)
            alpha = torch.where(mask[i].unsqueeze(1), alpha_next, alpha)

        return (alpha + self.end_transitions).logsumexp(dim=1)

    def decode(self, emissions, mask=None):
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.uint8, device=emissions.device)
        mask = mask.bool()

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        seq_length, batch_size, n_tags = emissions.shape
        alpha = self.start_transitions + emissions[0]
        history = []

        for i in range(1, seq_length):
            trans_score = self.transitions.unsqueeze(0).expand(batch_size, n_tags, n_tags)
            alpha_broadcast = alpha.unsqueeze(2).expand(batch_size, n_tags, n_tags)

            scores = alpha_broadcast + trans_score
            best_scores, best_tags = scores.max(dim=1)

            history.append(best_tags)
            alpha_next = best_scores + emissions[i]
            alpha = torch.where(mask[i].unsqueeze(1), alpha_next, alpha)

        alpha += self.end_transitions
        best_tags = alpha.argmax(dim=1)
        best_paths = [best_tags.unsqueeze(1)]

        for hist, m in zip(reversed(history), reversed(mask[1:])):
            best_prev = hist.gather(1, best_tags.unsqueeze(1))
            best_tags = torch.where(m.unsqueeze(1), best_prev, best_tags.unsqueeze(1)).squeeze(1)
            best_paths.insert(0, best_tags.unsqueeze(1))

        # FIX: The result is already (Batch, Seq), do not transpose!
        best_paths = torch.cat(best_paths, dim=1)

        return best_paths.tolist()