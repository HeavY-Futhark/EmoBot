# train_model.py continuation
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Assuming a function load_and_preprocess_data exists
from data.preprocess import load_and_preprocess_data

def initialize_transformers():
    model_names = [
        'roberta-large-nli-stsb-mean-tokens', 
        'distilbert-base-nli-mean-tokens', 
        'bert-large-nli-stsb-mean-tokens'
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
    for mlp in mlps:
        mlp.fit(X_train, y_train)

def evaluate_models(mlps, X_test, y_test):
    for idx, mlp in enumerate(mlps):
        y_pred = mlp.predict(X_test)
        print(f"Model {idx} Report:")
        print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    # Initialize transformers and MLPs
    transformers = initialize_transformers()
    mlps = create_mlp_models(transformers)

    # Load and preprocess your data
    # This function needs to be defined in preprocess.py
    X, y = load_and_preprocess_data('path/to/your/data.csv')

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Train each MLP with data
    train_models(mlps, X_train, y_train)

    # Evaluate models on the test set
    evaluate_models(mlps, X_test, y_test)

    # TODO: Add model saving functionality
    # save_models(mlps, 'models/mlp/')
