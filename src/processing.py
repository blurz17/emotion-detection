import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

NEGATIVE_WORDS = {
    "no", "not", "none", "neither", "never", "nobody", "nothing", "nowhere", 
    "without", "against", "negative", "deny", "refuse", "reject", "decline",
    "doesn't", "isn't", "wasn't", "shouldn't", "won't", "can't", "couldn't", 
    "don't", "haven't", "hasn't", "hadn't", "aren't", "weren't", "wouldn't", 
    "daren't", "needn't", "didn't", "doesnt", "isnt", "wasnt", "shouldnt", 
    "wont", "cant", "couldnt", "dont", "havent", "hasnt", "hadnt", "arent", 
    "weren't", "wouldnt", "darent", "neednt", "didnt"
}

for pkg in ["stopwords", "punkt", "punkt_tab", "wordnet"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

# Initialize once for speed
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english")) - NEGATIVE_WORDS


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
