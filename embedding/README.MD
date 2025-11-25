# Embedding Files

This folder should contain the embedding files used for the model.

## How to get Embeddings

You can download pre-trained FastText embeddings for the Uzbek language from the following sources:

1.  **Common Crawl Vectors:**
    *   Visit: [https://fasttext.cc/docs/en/crawl-vectors.html](https://fasttext.cc/docs/en/crawl-vectors.html) [[9]](https://fasttext.cc/docs/en/crawl-vectors.html)
    *   Search for "Uzbek".
    *   Download `cc.uz.300.vec.gz` or `cc.uz.300.bin.gz`.

2.  **Hugging Face (Facebook FastText Vectors):**
    *   Link: [https://huggingface.co/facebook/fasttext-uz-vectors](https://huggingface.co/facebook/fasttext-uz-vectors) [[4]](https://huggingface.co/facebook/fasttext-uz-vectors/tree/main)

**After downloading:**
1.  Extract the files if they are zipped (`.zip` or `.gz`).
2.  Place the `.vec` or `.bin` files in this `embedding/` directory.
3.  Update the `embed_file` path in your arguments or `w2v.py` if necessary.