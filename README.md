# Code and data for Aspect Based Sentiment Analysis with Gated Convolutional Networks


# Instructions:
Download glove or word2vec file and change the path in w2v.py correspondingly.
# Embedding Files

This folder should contain the embedding files used for the model.

# Embedding Files

This folder should contain the embedding files used for the model.

## How to get Embeddings

You can download pre-trained FastText embeddings for the Uzbek language from the following sources:

1.  **Common Crawl Vectors:**
    *   Visit: [https://fasttext.cc/docs/en/crawl-vectors.html](https://fasttext.cc/docs/en/crawl-vectors.html) [[9]](https://fasttext.cc/docs/en/crawl-vectors.html)
    *   Search for "Uzbek".
    *   Download `cc.uz.300.vec.gz` or `cc.uz.300.bin.gz`.
    
**After downloading:**
1.  Extract the files if they are zipped (`.zip` or `.gz`).
2.  Place the `.vec` or `.bin` files in this `embedding/` directory.
3.  Update the `embed_file` path in your arguments or `w2v.py` if necessary.
4. 
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
