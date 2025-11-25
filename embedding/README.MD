# Embedding Files

This folder should contain the embedding files used for the model.

## How to get Embeddings

You can download pre-trained FastText embeddings for the Uzbek language from the following sources:

1.  **Common Crawl Vectors:**
    *   Visit: [https://fasttext.cc/docs/en/crawl-vectors.html](https://fasttext.cc/docs/en/crawl-vectors.html) [[9]](https://fasttext.cc/docs/en/crawl-vectors.html)
    *   Search for "Uzbek".
    *   Download `cc.uz.300.vec.gz` and `cc.uz.300.bin.gz`.


**After downloading:**
1.  Extract the files if they are zipped (`.zip` or `.gz`).
2.  Place the `.vec` and `.bin` files in this `embedding/` directory.
3.  Update the `embed_file` path in your arguments or `w2v.py` if necessary.