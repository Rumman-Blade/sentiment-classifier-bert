# Sentiment Classifier with BERT

Fine-tuning BERT (`bert-base-uncased`) on the IMDB movie reviews dataset for binary sentiment classification (positive/negative) using HuggingFace Transformers.

Built as a deep-dive learning project to understand transformer-based NLP pipelines end to end.

---

## What This Project Does

- Fine-tunes a pretrained BERT model on 2,000 IMDB movie reviews
- Achieves ~84% accuracy on 500 held-out test samples (CPU baseline)
- Includes a clean eval script with per-sentence predictions and classification report
- Designed to scale up on Kaggle GPU with a single config change

---

## Project Structure
sentiment-classifier-bert/
├── configs/
│   └── config.yaml       # all hyperparameters in one place
├── data/                 # placeholder (dataset loads via HuggingFace)
├── src/
│   ├── train.py          # fine-tuning pipeline with HuggingFace Trainer
│   └── eval.py           # inference + metrics on custom sentences
├── outputs/              # saved model weights (gitignored)
├── requirements.txt
└── README.md

---

## Key Concepts Covered

- **BERT tokenization** — `[CLS]`, `[SEP]` tokens, attention masks, padding
- **Transfer learning** — pretrained BERT + classification head for downstream task
- **HuggingFace Trainer API** — training loop, evaluation, checkpointing
- **Evaluation metrics** — accuracy, F1 score, classification report
- **Config-driven training** — all settings in `config.yaml`, no hardcoded values

---

## How to Run

**1. Clone the repo and create a virtual environment:**
```bash
git clone https://github.com/Rumman-Blade/sentiment-classifier-bert
cd sentiment-classifier-bert
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Train:**
```bash
python src/train.py
```

**4. Evaluate:**
```bash
python src/eval.py
```

---

## Results

| Model | Dataset | Accuracy | F1 Score |
|-------|---------|----------|----------|
| BERT-base-uncased (fine-tuned) | IMDB (2k train / 500 test) | ~84% | ~84% |

> Full results on Kaggle GPU with complete dataset coming soon.

---

## What I Learned

- How BERT's encoder architecture differs from GPT-2's decoder
- Why `[CLS]` token is used for classification tasks
- How `warmup_steps` and `weight_decay` affect fine-tuning stability
- How the HuggingFace Trainer handles the full training loop internally
- How to evaluate NLP models beyond just accuracy (F1, classification report)

---

## Next Steps

- [ ] Scale up to full 25k IMDB dataset on Kaggle GPU
- [ ] Swap `bert-base-uncased` for `distilbert` and compare speed vs accuracy
- [ ] Add inference script for single sentence prediction
- [ ] Experiment with different `max_length` values

---

## Tech Stack

- Python 3.11
- PyTorch
- HuggingFace Transformers + Datasets
- scikit-learn
- BERT (bert-base-uncased)
