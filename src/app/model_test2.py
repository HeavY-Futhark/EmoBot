import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from datasets import load_dataset
import pandas as pd
# Create sentence embeddings using Sentence Transformers
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load the sem_eval_2018_task_1 dataset using datasets library
dataset = load_dataset('sem_eval_2018_task_1', 'subtask5.english')
df = pd.DataFrame(dataset['train'])

# Tokenizer for transforming text into input suitable for transformers
tokenizer = AutoTokenizer.from_pretrained("paraphrase-MiniLM-L6-v2")

# Encode the text and create sentence embeddings
def encode_text(text):
    tokens = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**tokens)
    return model_output.last_hidden_state.mean(dim=1)


# Encode each text in the dataset and store the embeddings
df['embeddings'] = df['Tweet'].apply(lambda x: model.encode(x))

