# End-to-End Aspect-Based Sentiment Analysis (ABSA) for Uzbek Language

This repository contains a state-of-the-art implementation of **End-to-End Aspect Based Sentiment Analysis** tailored for the **Uzbek language**. The system operates in a two-stage pipeline to strictly identify aspect terms and classify their sentiment polarity.

## 🧠 System Architecture

The solution uses a pipeline approach to ensure high accuracy and interpretability:

1.  **Stage 1: Aspect Term Extraction (ATE)**
    * **Model:** Bidirectional LSTM with a Conditional Random Field (BiLSTM-CRF).
    * **Goal:** Sequence labeling to identify precise aspect terms (e.g., "osh", "xizmat") using BIO tagging.
2.  **Stage 2: Sentiment Classification (ASC)**
    * **Model:** Gated Convolutional Networks (GCAE).
    * **Goal:** Predicts sentiment (Positive, Negative, Neutral) for the aspect terms extracted in Stage 1.

## 📌 Features

* **Uzbek Language Support**: Customized for Uzbek morphology using FastText (`cc.uz.300.vec`) embeddings.
* **Robust Extraction**: BiLSTM-CRF architecture ensures valid tag sequences (e.g., `I-ASP` cannot follow `O`).
* **Gated Sentiment Analysis**: GCAE model efficiently filters context to focus specifically on the target aspect.
* **Professional Logging**: Automated CSV logging, loss/accuracy visualization, and best model checkpointing.

## 🛠️ Installation

**Prerequisites:** Python 3.8+

### 1. Clone & Setup

```bash
git clone <your-repo-url>
cd E2E-uzbek-absa
```

### 2\. Create a Virtual Environment

**Windows:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
```

**Linux / Mac:**

```bash
python3.9 -m venv .venv
source .venv/bin/activate
```

### 3\. Install Dependencies

Install the required packages. This project uses a specific `requirements.txt` to ensure compatibility between PyTorch, CUDA, and TorchText.

```bash
pip install -r requirements.txt
```

## 📂 Project Structure

```text
.
├── data/
│   ├── absa/               # Aspect Extraction Data (BIO Format)
│   │   ├── train.json      # Training data for ATE
│   │   └── test.json       # Testing data for ATE
│   ├── asca/               # Aspect Category Datasets (JSON)
│   │   ├── acsa_train.json
│   │   └── acsa_test.json
│   └── asta/               # Aspect Term Datasets (JSON)
│       ├── asta_train.json
│       └── asta_test.json
├── embedding/
│   ├── cc.uz.300.vec       # Place your downloaded FastText vectors here
│   └── README.MD
├── model_files/
│   ├── train_ate.py        # Training script for Aspect Extraction
│   ├── dataset_ate.py      # Data loader for BIO format
│   ├── model_ate.py        # BiLSTM-CRF Architecture
│   ├── run.py              # Training script for Sentiment (GCAE)
│   ├── run_inference_ate.py # Inference script for Aspects
│   ├── cnn_train.py        # Training loop, validation, and logging logic
│   ├── cnn_gate_aspect_model.py # Neural network architecture
│   ├── inference.py        # CLI script for prediction
│   ├── visualize.py        # Script for generating analysis graphs
│   └── ...
├── snapshot/               # Stores snapshot of models and training logs with visualizations
└── requirements.txt
```

## 📊 Data Preparation

### 1\. Embeddings

You must download pre-trained Uzbek FastText vectors to initialize the model.

1.  Visit [fasttext.cc](https://fasttext.cc/docs/en/crawl-vectors.html).
2.  Search for **Uzbek** and download `cc.uz.300.vec.gz`.
3.  Extract the file and place `cc.uz.300.vec` inside the `embedding/` directory.

### 2\. Aspect Extraction Data (ATE)

Place your sequence labeling data in `data/absa/train.json`. Format:

```json
[
  {
    "tokens": ["Osh", "juda", "mazali", "edi"],
    "labels": ["B-ASP", "O", "O", "O"]
  }
]
```

### 3\. Sentiment Classification Data (ASC)

Place your sentiment data in `data/asca/acsa_train.json`. Format:

```json
[
  {
    "sentence": "Osh juda mazali edi.",
    "aspect": "food",
    "sentiment": "positive"
  },
  {
    "sentence": "Xizmat ko'rsatish sifati past.",
    "aspect": "service",
    "sentiment": "negative"
  }
]
```
## 🧩 Morphological-aware Tokenizer

As detailed in **Section 3.1** of the paper, this repository now supports **Morphologically-Informed Tokenization**. This addresses the agglutinative nature of Uzbek by segmenting words into valid morphemes (e.g., `dehqonchiliklaridan` -> `['dehqon', 'chilik', 'lari', 'dan']`) before feeding them into the model.

### Prerequisites
To use the FST-based tokenizer, you must install **Apertium-Uzbek**:

1.  **Install Apertium Core:**
    ```bash
    sudo apt-get install apertium apertium-dev
    ```
2.  **Install HFST:**
    ```bash
    pip install hfst
    ```
3.  **Download Uzbek Data:**
    Clone the `apertium-uzb` repository and compile:
    ```bash
    git clone [https://github.com/apertium/apertium-uzb.git](https://github.com/apertium/apertium-uzb.git)
    cd apertium-uzb
    ./autogen.sh && make
    ```
4.  **Configuration:**
    Point the tokenizer to your compiled `.automorf.bin` file in `model_files/morph_tokenizer.py` or place it in the `embedding/` folder.

### Usage
The `run.py` script is updated to use `morph_tokenize` by default. This will automatically segment input reviews during training and inference.

## 🚀 Usage

### 1\. Training the Model
### Stage 1: Train Aspect Extractor (BiLSTM-CRF)

Train the model to recognize aspect terms from raw text.

```bash
python model_files/train_ate.py --epochs 50 --batch_size 8 --lr 0.002
```

  * **Output:** Best model saved to `snapshot/ate_model/best_ate.pt`.


### Stage 2: To train the model, run the following command. By default, this runs the **ACSA** task.

```bash
python model_files/run.py -model CNN_Gate_Aspect -embed_file fasttext -epochs 10 -batch-size 32
```

  * **Outputs**:
      * **Best Model**: Saved to `snapshot/YYYY-MM-DD_HH-MM-SS/best_model.pt`.
      * **Training Log**: Saved to `snapshot/.../training_log.csv`.

**Training for Aspect Terms (ATSA):**
To train on the Aspect Term task, simply add the `-atsa` flag:

```bash
python model_files/run.py -model CNN_Gate_Aspect -atsa -embed_file fasttext -epochs 10
```

### 2\. Visualization & Analytics

After training is complete, you can generate professional graphs (Loss Curves, Accuracy Curves, Confusion Matrices) using the `visualize.py` script. You need to point it to the snapshot directory created during training.

```bash
# Replace the timestamp with your actual folder name
python model_files/visualize.py -log_dir "snapshot/2025-11-25_23-25-05"
```

  * **Generated Files**: `loss_curve.png`, `accuracy_curve.png`, and `confusion_matrix.png` will be saved inside the snapshot folder.

### 3\. Inference (Prediction)

To test your trained model on a single sentence, use the `inference.py` script.

**Example:**

```bash
python model_files/inference_triple.py -sentence "Osh juda yoqdi, xizmat sifati ham zo'r" -category "ovqat,xizmat" -term "osh,xizmat"
```
Arguments:
-sentence: The input text to analyze.
-category: Comma-separated list of aspect categories (e.g., "food,service").
-term: Comma-separated list of aspect terms (e.g., "osh,waiter").

```text
Result:
------------------------------------------------------------
TYPE       | TARGET               | PREDICTION           | CONF
------------------------------------------------------------
Category   | ovqat                | POSITIVE             | 98.2%
Category   | xizmat               | POSITIVE             | 92.3%
------------------------------------------------------------
Term       | osh                  | POSITIVE             | 98.6%
Term       | xizmat               | POSITIVE             | 95.9%
============================================================
```

  * **Note**: The `-aspect` argument must match one of the aspect categories present in your training data (e.g., `food`, `service`, `price`, `ambience`).

## ⚙️ Advanced Configuration

You can fine-tune the model using the following arguments in `run.py`:

| Argument | Default | Description |
| :--- | :--- | :--- |
| `-lr` | `0.01` | Initial learning rate. |
| `-dropout` | `0.5` | Dropout rate for regularization. |
| `-kernel-num` | `100` | Number of filters per kernel size. |
| `-kernel-sizes` | `3,4,5` | Comma-separated filter sizes for the CNN. |
| `-embed_file` | `w2v` | Embedding type (`w2v`, or `fasttext`). |
| `-l2` | `0` | L2 Regularization weight. |

## 📄 Citation

If you use this code or dataset in your research, please cite the following publication:

```bibtex
@inproceedings{YourName2025UzbekABSA,
  title={ [PLACEHOLDER FOR YOUR PAPER TITLE] },
  author={ [YOUR NAME] and [CO-AUTHORS] },
  booktitle={ [CONFERENCE/JOURNAL NAME] },
  year={2025}
}