import re
import numpy as np
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer
from sentence_transformers import SentenceTransformer

def preprocess_text(raw_post):
    # Configure stopwords set
    stops = set(stopwords.words("english"))

    # Remove non-letters and lower case
    letters_only = re.sub("[^a-zA-Z]", " ", raw_post)
    words = letters_only.lower().split()

    # Remove stopwords
    meaningful_words = [word for word in words if word not in stops]

    # Join the words back into one string
    clean_text = " ".join(meaningful_words)
    return clean_text

def preprocess_semeval_data(dataset, transformer_model):
    # Ensure the 'train' dataset is properly loaded and contains the 'text' field
    texts = [preprocess_text(example['text']) for example in dataset['train']]

    # Assuming you are working with a dataset that has emotion labels in a structured format
    # You need to adjust this part based on your specific dataset structure.
    # Here is an example assuming binary labels for each emotion category.
    emotions = []
    for example in dataset['train']:
        example_emotions = [
            example.get('anger', 0),
            example.get('anticipation', 0),
            example.get('disgust', 0),
            example.get('fear', 0),
            example.get('joy', 0),
            example.get('sadness', 0),
            example.get('surprise', 0),
            example.get('trust', 0)
        ]
        emotions.append(example_emotions)

    # Convert texts to embeddings using the provided transformer model
    text_embeddings = transformer_model.encode(texts)

    # Convert emotion labels into binary vectors if not already in that format
    # Assuming emotions is a list of lists with binary values
    return text_embeddings, np.array(emotions)
