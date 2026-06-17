import re
import string
import nltk
from nltk.corpus import stopwords
import contractions

stop_words = set(stopwords.words('english'))
negations = {
    'no',
    'nor',
    'not',
    "don't",
    "didn't",
    "won't",
    "wouldn't",
    "shouldn't",
    "couldn't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't"
}

important_words = {
    "too",
    "very",
    "more",
    "most",
    "less",
    "least"
}
stop_words = stop_words - negations - important_words

def clean_text(text):
    # convert to string (safety) and lowercase
    text = str(text)
    text = text.lower()

    # expand contractions
    text = contractions.fix(text)

    # remove HTML tags, URLs, numbers, punctuations, and extra whitespace
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    return text

def remove_stopwords(text):
    return " ".join([word for word in text.split() if word not in stop_words])

def preprocess_text(text):
    text = clean_text(text)
    text = remove_stopwords(text)
    return text