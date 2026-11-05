import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

try:
    from src.negation_words import NEGATIVE_WORDS, negative_words
except ImportError:
    from negation_words import NEGATIVE_WORDS, negative_words

# Ensure required NLTK data packages are available quietly
for pkg in ["stopwords", "punkt", "punkt_tab", "wordnet"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

# Initialize once for speed
lemmatizer = WordNetLemmatizer()
negation_set = NEGATIVE_WORDS if "NEGATIVE_WORDS" in locals() else set(negative_words)
stop_words = set(stopwords.words("english")) - negation_set


def clean_noise(text):
    """
    Remove URLs, special characters, and punctuation from text.
    """
    text = re.sub(r"http\S+", "", str(text))
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.strip()


def process_text(text):
    """
    Tokenize text, remove non-negative stopwords, and lemmatize words in lowercase.
    """
    cleaned = clean_noise(text)
    tokens = word_tokenize(cleaned)
    return [
        lemmatizer.lemmatize(w.lower())
        for w in tokens
        if w.lower() not in stop_words
    ]
