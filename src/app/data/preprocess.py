import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')  # For lemmatization

import spacy
#!python -m spacy download en_core_web_sm  # For Spacy's English model



import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy

# Load the Spacy model for more sophisticated tokenization and part-of-speech tagging.
nlp = spacy.load("en_core_web_sm")

def enhanced_preprocess_text(raw_text, use_lemmatization=True):
    # Configure stopwords set
    stops = set(stopwords.words("english"))

    # Remove non-letters and lower case
    letters_only = re.sub("[^a-zA-Z]", " ", raw_text)
    words = letters_only.lower().split()

    # Use Spacy for more sophisticated tokenization and lemmatization
    doc = nlp(' '.join(words))

    # Initialize WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    clean_words = []

    for word in doc:
        # Handling Stopwords
        if word.text not in stops:
            # Lemmatization (optional)
            if use_lemmatization:
                lemma = word.lemma_
                clean_words.append(lemma)
            else:
                clean_words.append(word.text)

    # Join the words back into one string
    clean_text = " ".join(clean_words)
    return clean_text

# Example usage:
text_example = "Here's an example sentence! Isn't it great?"
print(enhanced_preprocess_text(text_example))
