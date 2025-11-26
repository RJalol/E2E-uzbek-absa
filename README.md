Here is the updated, professional `README.md` documentation in Markdown format. I have removed the original paper reference and added the requested placeholder for your future publication.

-----

# Aspect Based Sentiment Analysis (ABSA) for Uzbek Language

This repository contains a comprehensive implementation of **Aspect Based Sentiment Analysis (ABSA)** specifically tailored for the **Uzbek language**, utilizing **Gated Convolutional Networks (GCAE)**. The project has been adapted to support Uzbek linguistic datasets, FastText embeddings, and includes a suite of professional analytics and inference tools.

The system is designed to handle two primary sub-tasks:

1.  **Aspect Category Sentiment Analysis (ACSA)**: Predicting sentiment (Positive, Negative, Neutral) for a specific aspect category (e.g., `food`, `service`, `price`) within a sentence.
2.  **Aspect Term Sentiment Analysis (ATSA)**: Predicting sentiment for specific aspect terms explicitly mentioned in the text.

## 📌 Features

  * **Gated Convolutional Networks**: Implements an efficient GCAE architecture optimized for aspect-specific sentiment extraction.
  * **Uzbek Language Support**: Fully integrated with Uzbek FastText (`cc.uz.300.vec`) embeddings and custom tokenization.
  * **Robust Training Pipeline**: Enhanced training loop featuring **Early Stopping**, **L2 Regularization**, and **Best Model Checkpointing** to prevent overfitting.
  * **Professional Analytics**: automated generation of training visualizations, including Loss/Accuracy curves and Confusion Matrices.
  * **Inference CLI**: A dedicated script for easy, interactive testing of the trained model on new sentences.
  * **Legacy Environment Compatibility**: Tuned to operate seamlessly with specific PyTorch/TorchText versions required for reproducibility.

## 🛠️ Installation

This project requires a specific Python environment to support the model architecture. It is highly recommended to use **Python 3.9**.

### 1\. Clone the Repository

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
│   ├── run.py              # Main entry point for training
│   ├── cnn_train.py        # Training loop, validation, and logging logic
│   ├── cnn_gate_aspect_model.py # Neural network architecture
│   ├── inference.py        # CLI script for prediction
│   ├── visualize.py        # Script for generating analysis graphs
│   └── ...
├── snapshot/               # Stores saved models and training logs
└── requirements.txt
```

## 📊 Data Preparation

### 1\. Embeddings

You must download pre-trained Uzbek FastText vectors to initialize the model.

1.  Visit [fasttext.cc](https://fasttext.cc/docs/en/crawl-vectors.html).
2.  Search for **Uzbek** and download `cc.uz.300.vec.gz`.
3.  Extract the file and place `cc.uz.300.vec` inside the `embedding/` directory.

### 2\. Datasets

Ensure your training and testing data is placed in `data/asca/` (for categories) or `data/asta/` (for terms). The files must be in **JSON format** with the following structure:

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

## 🚀 Usage

### 1\. Training the Model

To train the model, run the following command. By default, this runs the **ACSA** task.

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
python model_files/inference.py -snapshot "snapshot/2025-11-25_23-25-05/best_model.pt" -sentence "Osh juda mazali edi" -aspect "food"
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
| `-embed_file` | `w2v` | Embedding type (`w2v`, `glove`, or `fasttext`). |
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
```