#first model without transformers (DO NOT USE)
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
from tensorflow.keras.callbacks import History
from sklearn.metrics import hamming_loss
import re
from nltk.corpus import stopwords

def preprocess_text(raw_post):
    stops = set(stopwords.words("english"))

    # Remove anything that isn't a letter
    letters_only = re.sub("[^a-zA-Z]", " ", raw_post)

    # Convert to lowercase and split into words
    words = letters_only.lower().split()

    # Remove stopwords
    meaningful_words = [word for word in words if word not in stops]

    # Join words to form a clean string
    clean_text = " ".join(meaningful_words)

    return clean_text


"""
example_text = "This is an example text with some punctuation! And capitalization."
processed_text = preprocess_text(example_text)
print("Original Text:", example_text)
print("Processed Text:", processed_text)
"""



# Load the dataset
dataset = load_dataset('sem_eval_2018_task_1', 'subtask5.english')
print(dataset)
# Convert data into DataFrame
df = pd.DataFrame(dataset['train'])
# Preprocess the text
clean_posts = [preprocess_text(post) for post in df['Tweet']]

# Select emotion columns
emotion_columns = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']

# Use the emotion columns to create a new 'emotion' column
df['emotion'] = df[emotion_columns].idxmax(axis=1)
# Convert labels into numeric values
label_encoder = LabelEncoder()
df['emotion_encoded'] = label_encoder.fit_transform(df['emotion'])

# Define the vocabulary size
vocab_size = 10000

# Use Tokenizer to create sequences
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(clean_posts)

# Convert text into sequences
sequences = tokenizer.texts_to_sequences(clean_posts)

# Add padding so all sequences are of the same length
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, df['emotion_encoded'], test_size=0.2, random_state=42)

# Define the model
model = Sequential()
model.add(Embedding(vocab_size, 128, input_length=max_length))
model.add(LSTM(100))
model.add(Dense(11, activation='sigmoid'))

history = History()
# Compile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test),callbacks=[history])

# Evaluate the model
predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)
print(classification_report(y_test, y_pred))

# After training, print the training and validation loss history
print("Training Loss per Epoch:")
print(history.history['loss'])  # training loss

print("Validation Loss per Epoch:")
print(history.history['val_loss'])  # validation loss

# Calculate and print Hamming Loss
print("Hamming Loss:")
print(hamming_loss(y_test, y_pred))

