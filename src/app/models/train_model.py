# train_model.py continuation
from sentence_transformers import SentenceTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump  # For saving models
from datasets import load_dataset  # Import the load_dataset function
#from data.preprocess import preprocess_semeval_data


import numpy as np
from nltk.corpus import stopwords
import re
from sklearn.preprocessing import MultiLabelBinarizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm  # Import tqdm

def preprocess_text(raw_tweet):
    stops = set(stopwords.words("english"))
    letters_only = re.sub("[^a-zA-Z]", " ", raw_tweet)
    words = letters_only.lower().split()
    meaningful_words = [w for w in words if w not in stops]
    return " ".join(meaningful_words)

def preprocess_semeval_data(dataset, transformer_model):
    # Extract tweet texts
    texts = [preprocess_text(example['Tweet']) for example in dataset['train']]
    
    # Extract emotion labels
    emotion_labels = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']
    emotions = []
    for example in tqdm(dataset['train'], desc="Extracting Emotions"):
        example_emotions = [example[emotion] for emotion in emotion_labels]
        emotions.append(example_emotions)
       

    # Convert texts to embeddings using the provided transformer model
    text_embeddings = transformer_model.encode(texts, show_progress_bar=True)


    return text_embeddings, np.array(emotions)

# Other parts of the script remain the same...




def initialize_transformers():
    model_names = [
        'roberta-large-nli-stsb-mean-tokens'
    ]
    transformers = [SentenceTransformer(model) for model in model_names]
    return transformers

def create_mlp_models(transformers):
    mlps = []
    for _ in transformers:
        mlp = MLPClassifier(hidden_layer_sizes=(500, 300), activation='relu', solver='adam', random_state=1)
        mlps.append(mlp)
    return mlps

def train_models(mlps, X_train, y_train):
    print("training...")
    for idx, mlp in enumerate(tqdm(mlps, desc="Training Models")):
        print(f"Training MLP {idx+1}/{len(mlps)}...")
        mlp.fit(X_train, y_train)
        tqdm.write(f"Completed MLP {idx+1}/{len(mlps)}")

def evaluate_models(mlps, X_test, y_test):
    for idx, mlp in enumerate(mlps):
        y_pred = mlp.predict(X_test)
        print(f"Model {idx} Report:")
        print(classification_report(y_test, y_pred))

def save_models(mlps, save_dir):
    for idx, mlp in enumerate(mlps):
        model_path = f"{save_dir}/mlp_model_{idx+1}.joblib"
        dump(mlp, model_path)

if __name__ == "__main__":
    # Initialize transformers and MLPs
    transformers = initialize_transformers()
    mlps = create_mlp_models(transformers)

    # Load the SemEval dataset
    dataset = load_dataset('sem_eval_2018_task_1', 'subtask5.english')
    # Inspect the first example of the training set
    print(dataset['train'][0])

     # Choose one of the transformers as the model for creating embeddings
    transformer_model = transformers[0]  # For example, using the first one

    # Preprocess the data using the function from preprocess.py
    X, y = preprocess_semeval_data(dataset, transformer_model)

    print("preprocess ok")
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Train each MLP with data
    train_models(mlps, X_train, y_train)

    # Evaluate models on the test set
    evaluate_models(mlps, X_test, y_test)

    # Save the trained models
    save_models(mlps, 'saved_models')
