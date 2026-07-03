import yaml
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def tokenize_dataset(dataset, tokenizer, max_length: int):
    """
    Tokenize the entire dataset.
    
    Why map()? HuggingFace datasets are lazy — map() applies
    a function to every example efficiently in batches without
    loading everything into RAM at once.
    
    What tokenizer() does:
    - Adds [CLS] at start, [SEP] at end
    - Converts words to token IDs (integers)
    - Creates attention_mask: 1 for real tokens, 0 for padding
    - Pads/truncates to max_length
    """
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
    
    return dataset.map(tokenize, batched=True)


def compute_metrics(eval_pred):
    """
    Called by Trainer after each eval step.
    
    eval_pred is a tuple of (logits, labels):
    - logits: raw model outputs, shape (batch_size, num_labels)
    - labels: true labels, shape (batch_size,)
    
    We argmax the logits to get predicted class (0=negative, 1=positive)
    then compute accuracy and F1.
    
    Why F1 and not just accuracy? F1 is more robust when classes
    are imbalanced. IMDB is balanced (50/50) but F1 is good habit
    for any classification task.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    
    return {
        "accuracy": acc,
        "f1": f1,
    }


def main():
    # ── 1. Load config ────────────────────────────────────────────
    config = load_config("configs/config.yaml")
    print("Config loaded:", config)

    # ── 2. Load tokenizer ─────────────────────────────────────────
    print(f"\nLoading tokenizer: {config['model_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    # ── 3. Load dataset ───────────────────────────────────────────
    print("\nLoading IMDB dataset...")
    dataset = load_dataset(config["dataset_name"])

    # Use a subset for local dev (full dataset = 25k train, 25k test)
    train_dataset = dataset["train"].shuffle(seed=42).select(
        range(config["train_samples"])
    )
    test_dataset = dataset["test"].shuffle(seed=42).select(
        range(config["test_samples"])
    )
    print(f"Train size: {len(train_dataset)} | Test size: {len(test_dataset)}")

    # ── 4. Tokenize ───────────────────────────────────────────────
    print("\nTokenizing dataset...")
    train_tokenized = tokenize_dataset(train_dataset, tokenizer, config["max_length"])
    test_tokenized = tokenize_dataset(test_dataset, tokenizer, config["max_length"])

    # ── 5. Load model ─────────────────────────────────────────────
    # AutoModelForSequenceClassification adds a linear classification
    # head on top of BERT's [CLS] token output automatically.
    # num_labels=2 means binary classification (positive/negative)
    print(f"\nLoading model: {config['model_name']}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=2,
    )
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── 6. Training arguments ─────────────────────────────────────
    # TrainingArguments is a dataclass that holds ALL training settings.
    # The Trainer reads from this — no manual training loop needed.
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        warmup_steps=config["warmup_steps"],
        weight_decay=config["weight_decay"],
        logging_dir="outputs/logs",
        logging_steps=config["logging_steps"],
        eval_strategy="steps",
        eval_steps=config["eval_steps"],
        save_steps=config["save_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",           # disables wandb/tensorboard for now
    )

    # ── 7. Trainer ────────────────────────────────────────────────
    # Trainer handles: forward pass, loss, backward pass, optimizer
    # step, evaluation, checkpointing — all automatically.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=test_tokenized,
        compute_metrics=compute_metrics,
    )

    # ── 8. Train ──────────────────────────────────────────────────
    print("\nStarting training...")
    trainer.train()

    # ── 9. Final evaluation ───────────────────────────────────────
    print("\nRunning final evaluation...")
    results = trainer.evaluate()
    print(f"\nFinal Results:")
    print(f"  Accuracy: {results['eval_accuracy']:.4f}")
    print(f"  F1 Score: {results['eval_f1']:.4f}")

    # ── 10. Save model ────────────────────────────────────────────
    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    print(f"\nModel saved to {config['output_dir']}")


if __name__ == "__main__":
    main()