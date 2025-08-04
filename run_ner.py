import argparse
import os
from huggingface_hub import login, create_repo
from src.ner_model.data_loader import load_data, get_label_maps
from src.ner_model.model import load_model
from src.ner_model.tokenizer import NerTokenizer
from src.ner_model.trainer import NerTrainer, setup_training_args, setup_callbacks

def main(args):
    # --- Hugging Face Hub Login and Repo Creation ---
    if args.hf_token:
        login(token=args.hf_token)

    if args.push_to_hub:
        create_repo(args.repo_id, repo_type="model", exist_ok=True)

    # --- Load Data and Labels ---
    dataset = load_data(args.dataset_name)
    label_list, label2id, id2label = get_label_maps()

    # --- Initialize Tokenizer and Preprocess Data ---
    ner_tokenizer = NerTokenizer(hf_tokenizer_name=args.model_name)

    tokenized_datasets = dataset.map(
        lambda x: ner_tokenizer.tokenize_and_align_labels(x, label2id),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    # Split dataset
    split_datasets = tokenized_datasets["train"].train_test_split(test_size=0.1)
    train_dataset = split_datasets["train"]
    eval_dataset = split_datasets["test"]

    # --- Load Model ---
    model = load_model(model_name=args.model_name, num_labels=len(label_list))

    # --- Setup Trainer ---
    training_args = setup_training_args(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        early_stopping=args.early_stopping
    )

    callbacks = setup_callbacks(early_stopping=args.early_stopping)

    trainer = NerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=ner_tokenizer.tokenizer,
        label_list=label_list,
        callbacks=callbacks,
    )

    # --- Train and Evaluate ---
    trainer.train()
    eval_results = trainer.evaluate()
    print("--- Evaluation Results ---")
    print(eval_results)

    # --- Save and Push ---
    trainer.save_model(args.output_dir)
    if args.push_to_hub:
        trainer.push_to_hub(args.repo_id)
        print("Model pushed to hub successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a NER model with Sudachi.")

    # Model and Data Arguments
    parser.add_argument("--model_name", type=str, default="cl-nagoya/ruri-v3-130m", help="Base model name from Hugging Face Hub.")
    parser.add_argument("--dataset_name", type=str, default="stockmark/ner-wikipedia-dataset", help="Dataset name from Hugging Face Hub.")
    parser.add_argument("--output_dir", type=str, default="./ner_model_output", help="Directory to save model output.")

    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Training and evaluation batch size.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--early_stopping", action='store_true', help="Enable early stopping.")

    # Hugging Face Hub Arguments
    parser.add_argument("--push_to_hub", action='store_true', help="Push the model to the Hugging Face Hub.")
    parser.add_argument("--repo_id", type=str, help="Hugging Face Hub repository ID (e.g., your-username/your-repo-name).")
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"), help="Hugging Face API token.")

    args = parser.parse_args()

    if args.push_to_hub and not args.repo_id:
        raise ValueError("You must provide --repo_id when using --push_to_hub.")

    main(args)
