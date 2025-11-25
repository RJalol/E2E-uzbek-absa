# Code and data for Aspect Based Sentiment Analysis with Gated Convolutional Networks


# Instructions:
Download glove or word2vec file and change the path in w2v.py correspondingly.

## ACSA
python -m run -lr 1e-2 -batch-size 32  -verbose 1  -model CNN_Gate_Aspect    -embed_file glove  -r_l r  -epochs 13

python -m run -lr 1e-2 -batch-size 32  -verbose 1  -model CNN_Gate_Aspect    -embed_file glove  -r_l r  -year 14 -epochs 5

## ATSA
python -m run -lr 5e-3 -batch-size 32  -verbose 1  -model CNN_Gate_Aspect  -embed_file glove  -r_l r -year 14 -epochs 6 -atsa

python -m run -lr 5e-3 -batch-size 32  -verbose 1  -model CNN_Gate_Aspect  -embed_file glove  -r_l l -year 14 -epochs 5 -atsa


```
@inproceedings{
Soon here will be sited
}
```
