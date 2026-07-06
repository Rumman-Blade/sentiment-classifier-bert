import yaml
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def predict(model, tokenizer, texts: list, max_length: int, device: str) -> list:
    """
    Run inference on a list of texts.
    
    Why torch.no_grad()? During inference we don't need gradients
    — disabling them saves memory and speeds things up significantly.
    """
    model.eval()
    all_preds = []

    for text in texts:
        inputs = tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits                    # raw scores, shape (1, 2)
            pred = torch.argmax(logits, dim=-1).item() # 0=negative, 1=positive
            all_preds.append(pred)

    return all_preds


def main():
    config = load_config("configs/config.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running eval on: {device}\n")

    # ── Load fine-tuned model ─────────────────────────────────────
    print("Loading fine-tuned model from outputs/...")
    tokenizer = AutoTokenizer.from_pretrained(config["output_dir"])
    model = AutoModelForSequenceClassification.from_pretrained(config["output_dir"])
    model.to(device)

    # ── Custom test sentences ─────────────────────────────────────
    # These are sentences the model has NEVER seen during training
    test_sentences = [
        ("This movie was absolutely fantastic, I loved every minute!", 1),
        ("Terrible film, complete waste of time.", 0),
        ("The acting was decent but the plot was confusing.", 0),
        ("One of the best movies I have seen in years!", 1),
        ("I fell asleep halfway through, very boring.", 0),
        ("Brilliant performances from the entire cast.", 1),
        ("The special effects were impressive but the story was weak.", 0),
        ("A masterpiece of modern cinema.", 1),
    ]

    texts = [s[0] for s in test_sentences]
    true_labels = [s[1] for s in test_sentences]
    label_names = {0: "NEGATIVE", 1: "POSITIVE"}

    # ── Run predictions ───────────────────────────────────────────
    print("Running predictions on custom test sentences...\n")
    predictions = predict(model, tokenizer, texts, config["max_length"], device)

    # ── Print results ─────────────────────────────────────────────
    print("=" * 70)
    print("RESULTS: Fine-tuned BERT Sentiment Classifier")
    print("=" * 70)

    for text, true, pred in zip(texts, true_labels, predictions):
        status = "✓" if true == pred else "✗"
        print(f"{status} [{label_names[pred]}] {text[:60]}...")
        if true != pred:
            print(f"  ^ Expected: {label_names[true]}")

    # ── Metrics ───────────────────────────────────────────────────
    acc = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average="weighted")

    print("\n" + "=" * 70)
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("\nDetailed Report:")
    print(classification_report(true_labels, predictions,
                                target_names=["NEGATIVE", "POSITIVE"]))


if __name__ == "__main__":
    main()