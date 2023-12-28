print("ok")
import re
from datasets import load_dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Charger le dataset après le prétraitement
dataset = load_dataset("sem_eval_2018_task_1","subtask5.english")

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

dataset['train'] = [dict(item, Tweet=preprocess_text(item['Tweet'])) for item in dataset['train']]



# Créer une liste d'étiquettes d'émotion pour chaque exemple dans l'ensemble d'entraînement
train_labels = [
    [
        item['anger'],
        item['anticipation'],
        item['disgust'],
        item['fear'],
        item['joy'],
        item['love'],
        item['optimism'],
        item['pessimism'],
        item['sadness'],
        item['surprise'],
        item['trust']
    ]
    for item in dataset['train']
]

# Diviser le dataset en ensembles d'entraînement et de test
train_texts, test_texts, train_labels, test_labels = train_test_split(
    dataset['train']['Tweet'],
    train_labels,
    test_size=0.2,
    random_state=42
)

# Représentation du texte avec TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Choisissez un nombre approprié de fonctionnalités
X_train_tfidf = tfidf_vectorizer.fit_transform(train_texts)
X_test_tfidf = tfidf_vectorizer.transform(test_texts)

# Construction d'un modèle simple (Naive Bayes)
model = MultinomialNB()
model.fit(X_train_tfidf, train_labels)

# Prédictions sur l'ensemble de test
predictions = model.predict(X_test_tfidf)

# Évaluation du modèle
print("Accuracy :", accuracy_score(test_labels, predictions))
print("\nClassification Report :\n", classification_report(test_labels, predictions))
