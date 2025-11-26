import os
import sys
import time
import csv
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=3, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, args, epoch, save_path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, args, epoch, save_path, is_best=True)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, args, epoch, save_path, is_best=True)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, args, epoch, save_path, is_best=False):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        # Save Best Model (Generic Name)
        if is_best:
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pt'))
            self.val_loss_min = val_loss

        # Save Epoch Model
        filename = f"{args.model}_epoch_{epoch}.pt"
        torch.save(model.state_dict(), os.path.join(save_path, filename))


def train(train_iter, dev_iter, mixed_test_iter, model, args, text_field, aspect_field, sm_field, predict_iter):
    # Setup Logging
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    log_path = os.path.join(args.save_dir, 'training_log.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'time'])

    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.l2, lr_decay=args.lr_decay)

    # Initialize Early Stopping
    early_stopping = EarlyStopping(patience=5, verbose=True)

    steps = 0
    model.train()
    start_time = time.time()

    print(f"Training started. Logs will be saved to {log_path}")

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch in train_iter:
            feature, aspect, target = batch.text, batch.aspect, batch.sentiment

            # Data Preparation
            feature.t_()
            if len(feature) < 2: continue
            if not args.aspect_phrase:
                aspect.unsqueeze_(0)
            aspect.t_()
            target.sub_(1)

            if args.cuda:
                feature, aspect, target = feature.cuda(), aspect.cuda(), target.cuda()

            optimizer.zero_grad()
            logit, _, _ = model(feature, aspect)

            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            # Stats
            steps += 1
            batch_loss = loss.item()
            corrects = (torch.max(logit, 1)[1].view(target.size()) == target).sum().item()

            total_loss += batch_loss * batch.batch_size
            total_correct += corrects
            total_samples += batch.batch_size

            if steps % args.log_interval == 0:
                sys.stdout.write(
                    f'\rEpoch {epoch} | Step {steps} | Loss: {batch_loss:.4f} | Acc: {100.0 * corrects / batch.batch_size:.2f}%')

        # Epoch End: Calculate Train Metrics
        avg_train_loss = total_loss / total_samples
        avg_train_acc = 100.0 * total_correct / total_samples

        # Validation
        val_acc, val_loss, _ = eval(dev_iter, model, args)

        # Logging
        epoch_time = time.time() - epoch_start_time
        print(
            f"\nEpoch {epoch} Summary: Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Time: {epoch_time:.2f}s")

        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_train_loss, avg_train_acc, val_loss, val_acc, epoch_time])

        # Early Stopping & Checkpointing
        early_stopping(val_loss, model, args, epoch, args.save_dir)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Load best model for final testing
    print("Loading best model for final evaluation...")
    best_model_path = os.path.join(args.save_dir, 'best_model.pt')
    model.load_state_dict(torch.load(best_model_path))

    # Final Mixed Test
    mixed_acc = 0
    if mixed_test_iter:
        mixed_acc, _, _ = eval(mixed_test_iter, model, args)
        print(f"Final Mixed Test Accuracy: {mixed_acc:.2f}%")

    return (val_acc, mixed_acc), []  # Return format compatible with run.py


def eval(data_iter, model, args):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in data_iter:
            feature, aspect, target = batch.text, batch.aspect, batch.sentiment
            feature.t_()
            if not args.aspect_phrase:
                aspect.unsqueeze_(0)
            aspect.t_()
            target.sub_(1)

            if args.cuda:
                feature, aspect, target = feature.cuda(), aspect.cuda(), target.cuda()

            logit, pooling_input, relu_weights = model(feature, aspect)

            loss = F.cross_entropy(logit, target, reduction='sum')
            total_loss += loss.item()

            preds = torch.max(logit, 1)[1]
            total_correct += (preds.view(target.size()) == target).sum().item()
            total_samples += len(target)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0

    model.train()
    return accuracy, avg_loss, (all_preds, all_targets)