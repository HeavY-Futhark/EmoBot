import re
from gensim.models import Word2Vec
import numpy as np
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Charger le dataset à l'aide de la bibliothèque datasets
dataset = load_dataset("sem_eval_2018_task_1", "subtask5.english")

print(dataset)
example = dataset['train'][0]
print(example)

# Fonction de prétraitement du texte
def preprocess_text(text):
    # Supprime les caractères spéciaux et la ponctuation
    text = re.sub(r'[^\w\s]', '', text)

    text = text.lower()
    return text

# Applique le prétraitement à tous les textes dans l'ensemble d'entraînement
#dataset['train'] = [dict(item, Tweet=preprocess_text(item['Tweet'])) for item in dataset['train']]
# Appliquer le prétraitement à tous les textes dans l'ensemble d'entraînement
dataset['train'] = [dict(item, Tweet=preprocess_text(item['Tweet'])) for item in dataset['train']]


# Affiche quelques exemples avant et après le prétraitement
for i in range(5):
    print("Avant :", dataset['train'][i]['Tweet'])
    print("Après :", preprocess_text(dataset['train'][i]['Tweet']))
    print("-----")

# Statistiques avant le prétraitement
lengths_before = [len(item['Tweet'].split()) for item in dataset['train']]
print("Moyenne avant prétraitement :", np.mean(lengths_before))

# Statistiques après le prétraitement
lengths_after = [len(preprocess_text(item['Tweet']).split()) for item in dataset['train']]
print("Moyenne après prétraitement :", np.mean(lengths_after))

# Vérifie s'il y a des données manquantes après le prétraitement
missing_data_after = any('Tweet' not in item for item in dataset['train'])
print("Données manquantes après prétraitement :", missing_data_after)

# Convertir le dataset en DataFrame pandas
data = pd.DataFrame(dataset['train'])

# Prétraitement des données
# (Vous pouvez ajouter des étapes supplémentaires selon vos besoins)
text = data['Tweet']  # Utiliser 'Tweet' au lieu de 'text'
labels = data[['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']]

#Suppose train_texts, train_labels, test_texts, and test_labels are your data
# tokenized_texts = [text.split() for text in train_texts]

# Train a Word2Vec model
model_w2v = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)

# Function to average word vectors
def average_word_vectors(words, model, vocabulary, num_features):
    feature_vector = np.zeros((num_features,), dtype="float64")
    nwords = 0.

    for word in words:
        if word in vocabulary:
            nwords = nwords + 1.
            feature_vector = np.add(feature_vector, model.wv[word])

    if nwords:
        feature_vector = np.divide(feature_vector, nwords)

    return feature_vector

# Tokenize and build feature vectors for the training set
train_wordvec_arrays = np.zeros((len(tokenized_texts), 100), dtype="float64")

for i, text in enumerate(tokenized_texts):
    train_wordvec_arrays[i, :] = average_word_vectors(text, model_w2v, model_w2v.wv.index_to_key, 100)

# Tokenize and build feature vectors for the test set
tokenized_test_texts = [text.split() for text in test_texts]
test_wordvec_arrays = np.zeros((len(tokenized_test_texts), 100), dtype="float64")

for i, text in enumerate(tokenized_test_texts):
    test_wordvec_arrays[i, :] = average_word_vectors(text, model_w2v, model_w2v.wv.index_to_key, 100)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(train_wordvec_arrays, train_labels, test_size=0.2, random_state=42)

# Build a model with MultiOutputClassifier and linear SVM as base classifier
svm_model = MultiOutputClassifier(SVC(kernel='linear'))
svm_model.fit(X_train, y_train)

# Predictions on the test set
y_pred = svm_model.predict(X_test)

# Evaluation of the model
print("Accuracy :", accuracy_score(y_test, y_pred))
print("\nClassification Report :\n", classification_report(y_test, y_pred))

