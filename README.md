# To Vaccinate or Not to Vaccinate: Sentiment Analysis

This project compares sentiment classification approaches on the Zindi [To Vaccinate or Not to Vaccinate](https://zindi.africa/competitions/to-vaccinate-or-not-to-vaccinate-its-not-a-question) dataset. The task is to classify vaccine-related tweets as Negative (-1), Neutral (0), or Positive (1).

**Demo video:** [Watch the project demo](https://www.youtube.com/watch?v=VTsgU0uB_iE)

## Repository contents

```text
data/
├── raw/                 Original competition CSV files
└── processed/           Shared training and validation splits
notebooks/               EDA, five model approaches, and ablation experiments
figures/                 EDA and model evaluation plots
reports/
└── results/              Model metrics and ablation results in JSON
src/                     Shared preprocessing, datasets, models, and evaluation utilities
requirements.txt         Python dependencies
```

The notebooks cover exploratory analysis, TF-IDF with Logistic Regression and Linear SVM, a BiLSTM, CNN + BiGRU with attention, fine-tuned DistilBERT, model comparison, and ablation experiments.

## Setup

Install the project dependencies:

```bash
pip install -r requirements.txt
```

## Validation results

Scores below are on the held-out validation split of 1,449 tweets.

| Approach | Macro-F1 | Accuracy | ROC-AUC (macro, one-vs-rest) |
|---|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.675 | 0.727 | 0.864 |
| TF-IDF + Linear SVM | 0.665 | 0.732 | 0.850 |
| BiLSTM | 0.631 | 0.698 | 0.818 |
| CNN + BiGRU + Attention | 0.623 | 0.689 | 0.821 |
| Fine-tuned DistilBERT (1 epoch) | 0.591 | 0.672 | 0.855 |

DistilBERT had the lowest macro-F1 and the second-highest ROC-AUC, indicating a gap between its ranking ability and its class predictions at the chosen thresholds.

## Dataset observations

- The class distribution is imbalanced: Neutral 49%, Positive 41%, and Negative 10%.
- About 23% of duplicate-text groups have conflicting labels.
- The `#mmr` hashtag is often used in unrelated DJ and radio posts.
- User mentions, URLs, hashtags, and HTML entities are common text patterns.
- Train and test vocabularies differ substantially, with an estimated 40% out-of-vocabulary rate.
