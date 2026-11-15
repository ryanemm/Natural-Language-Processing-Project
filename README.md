# Researchly – Research Abstract Classification and Topic Analysis

ITNPA Natural Language Processing – Project 1.

The project has two parts:

1. **Research Papers Expert System (Questions 1–2).** A classifier that labels a research abstract as *artificial intelligence* or *classical* research, deployed as a Streamlit web app.
2. **Research Document Mining (Questions 3–4).** Denoising, summarisation (TextRank and TF-IDF) and LDA topic modelling of a small corpus of research articles, with an interactive pyLDAvis intertopic distance map.

## Repository structure

```
├── data/
│   ├── research_abstracts.json      # Q1–2: labelled research abstracts (ai / classical)
│   └── research_corpus/             # Q3–4: five research articles (research_1.txt … research_5.txt)
├── notebooks/
│   ├── question_1.ipynb             # denoising, feature engineering, baseline model
│   ├── question_2.ipynb             # model comparison, hyperparameter tuning, saves the deployed model
│   ├── question_3.ipynb             # article dataframe, denoising, TextRank and TF-IDF summaries
│   ├── question_4.ipynb             # LDA topics and pyLDAvis intertopic distance map
│   └── outputs/                     # all figures, result tables and the interactive map (.html)
├── deployment/                      # web app for the Q1–2 classifier – see deployment/README.md
├── documentation/
│   └── ITNPA4-34 … .pdf             # report: results, figures and commentary for Questions 1–4
└── requirements.txt
```

## Getting started

Python 3.11 or newer is recommended (the project was tested with Python 3.13).

```bash
git clone <repository-url>
cd <repository-folder>
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the notebooks

```bash
cd notebooks
jupyter notebook
```

Run the notebooks in order, from top to bottom. The notebooks are saved with their outputs, so they can also be read without running them.

| Notebook | Questions | Notes |
|---|---|---|
| `question_1.ipynb` | 1.1–1.3 | Uses the cleaning function in `deployment/preprocessing.py`. |
| `question_2.ipynb` | 2.1–2.3 | Writes the best model to `deployment/best_model.joblib`. Grid search takes a few minutes. |
| `question_3.ipynb` | 3.1–3.4 | Saves the cleaned corpus to `data/research_corpus_clean.pkl`. |
| `question_4.ipynb` | 4.1–4.2 | **Run `question_3.ipynb` first** – it uses the cleaned corpus. |

All figures and tables are saved to `notebooks/outputs/`. The interactive intertopic distance map (Q4.2) is `notebooks/outputs/q4_2_intertopic_distance_map.html`; open it in any web browser.

## Key results

- **Baseline (Q1):** TF-IDF (unigrams + bigrams) with class-balanced Logistic Regression – test macro F1 0.862, accuracy 90.7%.
- **Best model (Q2):** tuned Logistic Regression – cross-validated macro F1 0.866, test macro F1 0.887, accuracy 92.9%. This model is deployed.
- **Topic analysis (Q3–4):** the corpus is about research methodology (triangulation, reflexivity, instructional design, Cronbach's alpha, mixed methods) rather than AI.

## Deployment

The web interface for the classifier is in [`deployment/`](deployment/). See [`deployment/README.md`](deployment/README.md) for how to run it locally or deploy it to Streamlit Community Cloud.

## Documentation

[`documentation/`](documentation/) contains the report (PDF) with the results, figures and commentary for Questions 1–4.
