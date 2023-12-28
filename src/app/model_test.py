from datasets import load_dataset
import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam


import re
from nltk.corpus import stopwords

def preprocess_text(raw_post):
    stops = set(stopwords.words("english"))

    # Suppression de tout ce qui n'est pas une lettre
    letters_only = re.sub("[^a-zA-Z]", " ", raw_post)

    # Conversion en minuscules + Split des mots
    words = letters_only.lower().split()

    # Suppression des stopwords
    meaningful_words = [word for word in words if word not in stops]

    # Rejoindre les mots pour former une chaîne de nouveau
    clean_text = " ".join(meaningful_words)

    return clean_text



example_text = "This is an example text with some punctuation! And capitalization."
processed_text = preprocess_text(example_text)
print("Original Text:", example_text)
print("Processed Text:", processed_text)




# Charger le dataset
dataset = load_dataset('sem_eval_2018_task_1', 'subtask5.english')
print(dataset)
# Convertir les données en DataFrame
df = pd.DataFrame(dataset['train'])
# Prétraitement du texte
clean_posts = [preprocess_text(post) for post in df['Tweet']]

# Sélection des colonnes d'émotions
emotion_columns = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']

# Utiliser les colonnes d'émotions pour créer une nouvelle colonne 'emotion'
df['emotion'] = df[emotion_columns].idxmax(axis=1)
# Convertir les étiquettes en valeurs numériques
label_encoder = LabelEncoder()
df['emotion_encoded'] = label_encoder.fit_transform(df['emotion'])

# Définir la taille du vocabulaire
vocab_size = 10000

# Utiliser Tokenizer pour créer des séquences
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(clean_posts)

# Convertir le texte en séquences
sequences = tokenizer.texts_to_sequences(clean_posts)

# Ajouter du rembourrage pour que toutes les séquences aient la même longueur
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, df['emotion_encoded'], test_size=0.2, random_state=42)

# Définir le modèle
model = Sequential()
model.add(Embedding(vocab_size, 128, input_length=max_length))
model.add(LSTM(100))
model.add(Dense(8, activation='sigmoid'))

# Compiler le modèle
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Évaluer le modèle
predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)
print(classification_report(y_test, y_pred))
