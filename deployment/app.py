"""Question 2.3 - Researchly Research Abstract Classifier (web interface).

best_model.joblib is created by notebooks/question_2.ipynb (already included
in this folder). Start the app from this folder with:

    streamlit run app.py
"""
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

from preprocessing import clean_text

MODEL_PATH = Path(__file__).parent / 'best_model.joblib'
LABEL_NAMES = {'ai': 'artificial intelligence', 'classical': 'classical research'}

st.set_page_config(page_title='Researchly Abstract Classifier', page_icon='📄')


@st.cache_resource
def load_model():
    """Load the saved model once and reuse it for every prediction."""
    return joblib.load(MODEL_PATH)


def class_probabilities(pipeline, cleaned_text):
    """Return {label: probability} if the model supports it, else None."""
    if hasattr(pipeline, 'predict_proba'):
        probs = pipeline.predict_proba([cleaned_text])[0]
        return dict(zip(pipeline.classes_, probs))
    return None


st.title('Researchly Research Abstract Classifier')
st.write('This app classifies research abstracts as either AI-related or classical research.')

if not MODEL_PATH.exists():
    st.error('Model file not found. Run notebooks/question_2.ipynb to train '
             'and save the best model, then restart the app.')
    st.stop()

bundle = load_model()
pipeline = bundle['pipeline']

# Model details in the sidebar, so testers know which configuration they use
with st.sidebar:
    st.header('Deployed model')
    st.write(f"**{bundle['model_name']}**")
    st.write(f"Trained on {bundle['n_train']} abstracts")
    st.metric('Cross-validated macro F1', f"{bundle['cv_macro_f1']:.3f}")
    st.metric('Test macro F1', f"{bundle['test_macro_f1']:.3f}")
    st.metric('Test accuracy', f"{bundle['test_accuracy']:.1%}")
    if isinstance(bundle['params'], dict):
        st.caption('Tuned hyperparameters')
        st.json(bundle['params'])

abstract = st.text_area('Enter research abstract:', height=220)

if st.button('Classify Abstract'):
    cleaned = clean_text(abstract)

    if not abstract.strip():
        st.warning('Please enter an abstract to classify.')
    elif not cleaned:
        st.warning('The abstract has no meaningful words left after cleaning '
                   '(for example, only numbers or stopwords). Please enter more text.')
    else:
        prediction = pipeline.predict([cleaned])[0]

        st.subheader('Prediction')
        st.success(f'The abstract is classified as: {LABEL_NAMES.get(prediction, prediction)}')

        probs = class_probabilities(pipeline, cleaned)
        if probs is not None:
            st.write(f"Confidence: **{probs[prediction]:.1%}**")
            for label in pipeline.classes_:
                st.progress(float(probs[label]),
                            text=f'{LABEL_NAMES.get(label, label)}: {probs[label]:.1%}')

        n_known = len(set(cleaned.split()) & set(
            pipeline.named_steps['tfidf'].vocabulary_))
        if n_known < 5:
            st.info('Few words in this abstract were seen during training, so the '
                    'prediction may be unreliable.')

        with st.expander('Show preprocessed text'):
            st.write(cleaned)
