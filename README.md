# Japanese NER Model Training with ModernBERT and Sudachi

This project provides a reusable Python module for training a Named Entity Recognition (NER) model for Japanese text. It is a refactored and modularized version of the original `modernBERT_JA_NER_Sudachi.ipynb` notebook.

The model uses a ModernBERT-based architecture (`cl-nagoya/ruri-v3-130m`) and the Sudachi tokenizer for morphological analysis, which is crucial for handling Japanese text.

## Features

- **Modular Design**: Code is organized into logical modules for data loading, tokenization, model handling, and training.
- **Configurable**: Training parameters, model names, and dataset names can be easily configured via command-line arguments.
- **Reproducible**: `requirements.txt` ensures a consistent environment.
- **Hugging Face Hub Integration**: Easily push your trained models to the Hugging Face Hub.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Set Hugging Face Token (Optional):**
    If you plan to push the model to the Hugging Face Hub, you need to provide your API token. You can do this by setting an environment variable:
    ```bash
    export HF_TOKEN="your_huggingface_token_here"
    ```
    Alternatively, you can pass the token directly using the `--hf_token` argument.

## How to Run

The `run_ner.py` script is the main entry point for training the model.

### Command-Line Arguments

- `--model_name`: (str) The base model name from the Hugging Face Hub. Default: `cl-nagoya/ruri-v3-130m`.
- `--dataset_name`: (str) The dataset name from the Hugging Face Hub. Default: `stockmark/ner-wikipedia-dataset`.
- `--output_dir`: (str) The directory where the trained model will be saved. Default: `./ner_model_output`.
- `--epochs`: (int) The number of training epochs. Default: `3`.
- `--batch_size`: (int) The batch size for training and evaluation. Default: `16`.
- `--learning_rate`: (float) The learning rate. Default: `2e-5`.
- `--weight_decay`: (float) The weight decay. Default: `0.01`.
- `--early_stopping`: (flag) Use this flag to enable early stopping.
- `--push_to_hub`: (flag) Use this flag to push the final model to the Hugging Face Hub.
- `--repo_id`: (str) The repository ID on the Hugging Face Hub (e.g., `YourUsername/YourRepoName`). **Required if `--push_to_hub` is used.**
- `--hf_token`: (str) Your Hugging Face API token.

### Example Usage

**Basic Training:**
This command trains the model for 5 epochs and saves it locally to the `ner_model_output` directory.

```bash
python run_ner.py --epochs 5
```

**Training with Early Stopping and Pushing to Hub:**
This command enables early stopping and pushes the best model to the specified repository on the Hugging Face Hub.

```bash
python run_ner.py \
    --epochs 50 \
    --early_stopping \
    --push_to_hub \
    --repo_id "YourUsername/my-japanese-ner-model"
```
