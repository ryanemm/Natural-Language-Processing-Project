"""Text denoising shared by Question 2 (training) and the web app (2.3).

Using the exact same cleaning function at training time and at prediction time
guarantees that text typed into the web interface is transformed in the same
way as the abstracts the model was trained on.
"""
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Publisher boilerplate, e.g. "Copyright © 2011 John Wiley & Sons, Ltd."
COPYRIGHT_RE = re.compile(r'(copyright\s*)?©.*$|copyright\s+\d{4}.*$',
                          flags=re.IGNORECASE)
# Structured-abstract section headers, e.g. "Purpose-", "Summary", "Abstract:"
HEADER_RE = re.compile(
    r'\b(abstract|summary|purpose|design/methodology/approach|findings|'
    r'originality/value|research limitations/implications|practical implications)'
    r'\s*[-:–]', flags=re.IGNORECASE)


def clean_text(text):
    """Denoise an abstract: same steps as Question 1.1."""
    if not isinstance(text, str):
        return ""
    text = COPYRIGHT_RE.sub(' ', text)
    text = HEADER_RE.sub(' ', text)
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'<.*?>|&[a-z]+;', ' ', text)
    text = re.sub(r'[-‐–—/]', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = [lemmatizer.lemmatize(w) for w in text.split()
             if w not in stop_words and len(w) > 2]
    return " ".join(words)
