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
