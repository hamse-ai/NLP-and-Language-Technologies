# To Vaccinate or Not to Vaccinate — Sentiment Analysis

Formative Assignment 2 for Research-Informed Sequential Models for NLP and Language Technologies. Group project using the Zindi challenge
["To Vaccinate or Not to Vaccinate"](https://zindi.africa/competitions/to-vaccinate-or-not-to-vaccinate-its-not-a-question).

Our question: how well do sequential modelling approaches actually hold up on a real vaccine-hesitancy sentiment task, and what does the evidence say about where each approach is strong or weak?

The task itself is simple to state: classify vaccine-related tweets as Negative (-1), Neutral (0), or Positive (1).

**GitHub repository:** https://github.com/hamse-ai/NLP-and-Language-Technologies
**Demo video (7-10 min):** [INSERT DEMO VIDEO LINK]
**Group contribution tracker:** [INSERT CONTRIBUTION TRACKER LINK]

## Project structure
```
.
├── data/
│   ├── raw/                    # Train.csv, Test.csv, SampleSubmission.csv
│   └── processed/               # Shared stratified train/val split (train_split.csv, val_split.csv)
├── notebooks/
│   ├── eda.ipynb              # Exploratory data analysis (Section 1)
│   ├── baselines.ipynb        # Approach 1-2: TF-IDF + LogReg / Linear SVM
│   ├── bilstm.ipynb           # Approach 3: BiLSTM
│   ├── cnn_bigru_attention.ipynb  # Approach 4: CNN + BiGRU + Attention
│   ├── distilbert_finetuned.ipynb # Approach 5: fine-tuned DistilBERT
│   ├── results_comparison.ipynb   # Cross-model comparison + error analysis (Section 4)
│   └── ablation_experiments.ipynb # Hyperparameter/design-decision ablations (Section 3)
├── figures/                    # PNG figures exported from all notebooks
├── reports/
│   ├── eda_summary.md            # Written EDA findings + modelling implications
│   ├── related_work.md           # Section 2: literature review + model-selection rationale
│   ├── report.md                 # Full assembled report (source for the final PDF)
│   ├── Formative2_Report.pdf     # Generated locally; PDF artifacts are not committed
│   ├── results/                  # Per-model metrics as JSON (macro-F1, confusion matrix, ...)
│   │   └── ablations/            # Ablation-experiment results (notebooks/ablation_experiments.ipynb)
├── src/                         # Shared preprocessing, datasets, models, training/eval code
├── NLP_Primer_twitter_challenge.ipynb   # Reference-only starter notebook from the original
│                                         # Zindi hackathon (fastai/simpletransformers) — not
│                                         # one of our 5 approaches, not executed/graded by us
└── requirements.txt
```

## Setup
```bash
pip install -r requirements.txt
python src/data_split.py   # regenerate the shared train/val split
```
We ran notebooks through `nbclient` rather than the `jupyter nbconvert` CLI, which didn't work in our environment. Any notebook shows the pattern, or you can run:
```bash
python -c "import nbformat; from nbclient import NotebookClient; nb = nbformat.read('notebooks/baselines.ipynb', as_version=4); NotebookClient(nb, timeout=1800, kernel_name='python3', resources={'metadata': {'path': 'notebooks'}}).execute(); nbformat.write(nb, 'notebooks/baselines.ipynb')"
```
To rebuild the PDF after editing `reports/report.md`:
```bash
python scripts/build_pdf.py
```
Everything here ran locally on CPU — no GPU, 4 threads — and we haven't tried it on Colab. The notebooks use relative paths from inside `notebooks/` (things like `../data/processed/...`), so running on Colab would mean cloning the repo there and keeping that folder structure. The DistilBERT notebook in particular would go a lot faster with a GPU.

## Progress
- [x] **1. Problem & Data Investigation** — `notebooks/eda.ipynb`, `reports/eda_summary.md`
- [x] **2. Related Work & Sequential Model Selection** — `reports/related_work.md`
- [x] **3. Experimental Investigation** — all 5 approaches implemented and evaluated (see `notebooks/`), plus 7 real ablation experiments (three model-setting comparisons and four robustness checks) (`notebooks/ablation_experiments.ipynb`)
- [x] **4. Error Analysis & Limitations** — `notebooks/results_comparison.ipynb`
- [x] Final report — generated locally from `reports/report.md`; PDF output is intentionally excluded from the repository
- [ ] Demo video — script ready at `reports/demo_video_script.md`, recording is a group action
- [ ] Group contribution tracker — filled in at `reports/contribution_tracker.docx` from the group's `Task_Division.pdf`; still needs real emails/dates and the meeting log
- [ ] Push this repo to a GitHub remote (needs a group member's GitHub account/credentials)

## Results (held-out validation split, 1,449 tweets)
| Approach | Macro-F1 | Accuracy | ROC-AUC (macro, one-vs-rest) |
|---|---|---|---|
| TF-IDF + Logistic Regression | 0.675 | 0.727 | 0.864 |
| TF-IDF + Linear SVM | 0.665 | 0.732 | 0.850 |
| BiLSTM | 0.631 | 0.698 | 0.818 |
| CNN + BiGRU + Attention | 0.623 | 0.689 | 0.821 |
| Fine-tuned DistilBERT (1 epoch) | 0.591 | 0.672 | 0.855 |

The interesting bit is DistilBERT: worst macro-F1 of the five, but second-best ROC-AUC. That gap is basically the signature of a model that's undertrained (we only had the CPU budget for 1 epoch) rather than one that's just a bad fit for the task. The full writeup — discussion, error analysis, limitations, and all seven ablation experiments — is in `reports/Formative2_Report.pdf`, Sections 4.8-7.

## What the EDA told us (details in `reports/eda_summary.md`)
- Classes are imbalanced — Neutral 49%, Positive 41%, Negative only 10%.
- 23% of duplicate-text groups have conflicting labels. That's real label noise, not just annotators disagreeing.
- `#mmr` is the most common hashtag, but ~95% of those tweets are unrelated DJ/radio content that happens to share the string "MMR" — worth checking specifically during error analysis.
- `<user>`, `<url>`, hashtags, and un-decoded HTML entities show up constantly and need explicit handling.
- A 40% OOV rate between train and test, plus a long tail of words that only appear once, is a good argument for subword tokenization over a fixed vocabulary.
- All three classes share pretty much the same core vocabulary — sentiment comes from word order and framing, not which words show up, which is why sequence-aware models make sense over bag-of-words here.
