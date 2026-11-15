# Deployment – Researchly Research Abstract Classifier

A Streamlit web app that classifies a research abstract as **artificial intelligence** or **classical research** (Questions 1–2).

The deployed model is the best configuration from Question 2: TF-IDF (unigrams + bigrams) with a tuned, class-balanced Logistic Regression.

| Metric | Score |
|---|---|
| Cross-validated macro F1 | 0.866 |
| Test macro F1 | 0.887 |
| Test accuracy | 92.9% |

## Files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit web interface |
| `best_model.joblib` | The trained pipeline (TF-IDF vectoriser + classifier), created by `notebooks/question_2.ipynb` |
| `preprocessing.py` | Text-cleaning function, shared with the training notebooks so user input is cleaned exactly like the training data |
| `requirements.txt` | Packages needed to run the app |

## Run locally (3 steps)

From the repository root:

```bash
cd deployment
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser at http://localhost:8501. On the first run, NLTK downloads its stopword and WordNet data automatically, which needs an internet connection.

## Using the app

1. Paste a research abstract into the text box.
2. Click **Classify Abstract**.
3. The prediction appears below, with the model's confidence for each class.

The sidebar shows the deployed model and its scores. Expand **Show preprocessed text** to see the abstract after cleaning.

## Deploy online (Streamlit Community Cloud)

1. Push this repository to GitHub.
2. Sign in at https://share.streamlit.io with your GitHub account and click **Create app**.
3. Select the repository and branch, and set **Main file path** to `deployment/app.py`.
4. Click **Deploy**. Streamlit installs the packages from `deployment/requirements.txt` and gives the app a public URL.

## Retraining the model

Run `notebooks/question_2.ipynb` from top to bottom. Its last cell saves the best configuration to `deployment/best_model.joblib`; restart the app to use the new model.

## Troubleshooting

- **"Model file not found"** – `best_model.joblib` is missing from this folder; run `notebooks/question_2.ipynb` to create it.
- **Warnings about an inconsistent scikit-learn version** – the model was saved with scikit-learn 1.7.2; install that version (`pip install scikit-learn==1.7.2`) or retrain the model.
