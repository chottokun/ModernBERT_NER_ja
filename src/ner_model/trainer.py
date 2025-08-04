import numpy as np
import evaluate
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    get_linear_schedule_with_warmup,
)
from transformers.trainer_callback import EarlyStoppingCallback

def compute_metrics(eval_preds, label_list: list) -> dict:
    """
    Computes precision, recall, F1, and accuracy for NER.
    """
    metric = evaluate.load("seqeval")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Convert predictions and labels to string format
    true_labels = [[label_list[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)

    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }

def setup_training_args(
    output_dir,
    num_train_epochs,
    per_device_train_batch_size,
    per_device_eval_batch_size,
    weight_decay,
    learning_rate,
    early_stopping
) -> TrainingArguments:
    """
    Creates and returns a TrainingArguments object.
    """
    return TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        load_best_model_at_end=early_stopping,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
    )

def setup_callbacks(early_stopping: bool) -> list:
    """
    Configures and returns a list of callbacks.
    """
    callbacks = []
    if early_stopping:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0))
    return callbacks

class NerTrainer:
    def __init__(
        self,
        model,
        args: TrainingArguments,
        train_dataset,
        eval_dataset,
        tokenizer,
        label_list,
        callbacks
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.label_list = label_list

        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        num_training_steps = len(train_dataset) // args.per_device_train_batch_size * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )

        self.trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorForTokenClassification(tokenizer),
            compute_metrics=lambda p: compute_metrics(p, self.label_list),
            tokenizer=tokenizer,
            optimizers=(optimizer, scheduler),
            callbacks=callbacks,
        )

    def train(self):
        print("Starting training...")
        self.trainer.train()

    def evaluate(self):
        print("Evaluating model...")
        return self.trainer.evaluate()

    def save_model(self, output_dir: str):
        print(f"Saving model to {output_dir}...")
        self.trainer.save_model(output_dir)

    def push_to_hub(self, repo_id: str):
        print(f"Pushing model to Hugging Face Hub repository '{repo_id}'...")
        self.trainer.push_to_hub(repo_id)
