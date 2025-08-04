# Design for NER Model Module

This document outlines the design for refactoring the `modernBERT_JA_NER_Sudachi.ipynb` notebook into a reusable Python module.

## 1. Project Goals

- **Modularity**: Break down the monolithic notebook into logical, reusable components.
- **Reusability**: Create a package that can be easily used for training and evaluating NER models with different configurations.
- **Maintainability**: Improve code organization and readability for easier future development.
- **Configurability**: Allow users to easily change parameters like model names, hyperparameters, and dataset names without modifying the source code.

## 2. File Structure

The project will be organized into a `src` directory containing the main Python package and a root-level script for execution.

```
.
├── .gitignore
├── modernBERT_JA_NER_Sudachi.ipynb
├── requirements.txt
├── design.md
├── README.md
├── run_ner.py                  # Main execution script
└── src/
    └── ner_model/
        ├── __init__.py
        ├── data_loader.py          # Handles dataset and label loading
        ├── tokenizer.py            # Manages Sudachi/HF tokenization and label alignment
        ├── model.py                # Handles model loading
        └── trainer.py              # Manages the training and evaluation process
```

## 3. Module Breakdown

### 3.1. `run_ner.py` (Main Script)

This script is the entry point for the entire pipeline.

- **Responsibilities**:
    - Parse command-line arguments for configuration (e.g., model name, dataset, epochs, batch size, Hugging Face repo ID).
    - Orchestrate the training flow by calling functions/classes from the `ner_model` package.
    - Handle Hugging Face Hub login and model pushing.
- **Key Components**:
    - Use Python's `argparse` library for command-line argument handling.
    - A `main()` function to encapsulate the high-level logic.

### 3.2. `src/ner_model/data_loader.py`

This module is responsible for all data loading tasks.

- **Key Functions**:
    - `load_data(dataset_name: str)`: Loads a dataset from the Hugging Face Hub. Returns a `datasets.DatasetDict`.
    - `get_label_maps()`: Returns the `label_list`, `label2id`, and `id2label` dictionaries required for the NER task. For this project, it will be specific to the `stockmark/ner-wikipedia-dataset`.

### 3.3. `src/ner_model/tokenizer.py`

This module contains the complex logic for tokenization and aligning NER labels between word-level and subword-level tokens.

- **Key Class**: `NerTokenizer`
    - `__init__(self, hf_tokenizer_name: str)`:
        - Initializes the Hugging Face `AutoTokenizer`.
        - Initializes the `SudachiPy` tokenizer.
    - `sudachi_tokenizer(self, text: str) -> list[str]`:
        - A helper method to tokenize text into words using Sudachi.
    - `tokenize_and_align_labels(self, examples: dict, label2id: dict) -> dict`:
        - This is the core method that replicates the `tokenize_and_align_labels_sudachi` function from the notebook.
        - It performs Sudachi tokenization, Hugging Face subword tokenization, and uses `spacy-alignments` to correctly map labels to subwords.
        - Returns a dictionary containing `input_ids`, `attention_mask`, and `labels`.

### 3.4. `src/ner_model/model.py`

A simple module for loading the pre-trained model.

- **Key Function**:
    - `load_model(model_name: str, num_labels: int)`:
        - Loads an `AutoModelForTokenClassification` from the specified `model_name`.
        - Configures the model head with the correct number of labels.
        - Returns the model object.

### 3.5. `src/ner_model/trainer.py`

This module encapsulates the logic related to the `transformers.Trainer`.

- **Key Class**: `NerTrainer`
    - `__init__(self, model, training_args, train_dataset, eval_dataset, tokenizer, data_collator, compute_metrics, callbacks)`:
        - Initializes the `transformers.Trainer`.
    - `train(self)`:
        - Starts the training process.
    - `evaluate(self)`:
        - Runs evaluation on the test set.
    - `save_model(self, output_dir: str)`:
        - Saves the final model to a local directory.
    - `push_to_hub(self, repo_id: str)`:
        - Pushes the model and tokenizer to the specified Hugging Face Hub repository.
- **Helper Functions**:
    - `compute_metrics(eval_preds, label_list: list) -> dict`:
        - A standalone function to calculate performance metrics (precision, recall, F1, accuracy) using `seqeval`.
    - `setup_training_args(**kwargs) -> TrainingArguments`:
        - A function to create and return a `TrainingArguments` object based on provided arguments.
    - `setup_callbacks(early_stopping: bool) -> list`:
        - A function to configure and return a list of callbacks, such as `EarlyStoppingCallback`.
