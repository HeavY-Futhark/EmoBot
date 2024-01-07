from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
import os
import zipfile
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

# Set of English stopwords
stops = set(stopwords.words("english"))
# Function to ensure model is unzipped
def ensure_model_is_unzipped(model_dir):
    # Paths for the model and zip file within the 'saved_model' directory
    model_path = os.path.join(model_dir, "pytorch_model.bin")
    zip_path = os.path.join(model_dir, "pytorch_model.zip")  # Adjusted to .zip

    if not os.path.isfile(model_path):
        if os.path.isfile(zip_path):
            try:
                print(f"Attempting to unzip model from {zip_path}")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(model_dir)
                print("Model unzipped successfully.")
            except zipfile.BadZipFile:
                print(f"The file at {zip_path} is not a valid zip file. Please replace it with a valid .zip model file.")
        else:
            print(f"No model found at {model_path} or {zip_path}. Please ensure the model files are placed correctly.")
    else:
        print("Model is already unzipped.")


def preprocess_text(raw_text):
    """
    Perform cleaning and standardization on the input text, with a simplistic approach to negation.

    Parameters:
    raw_text (str): The text to be preprocessed.

    Returns:
    str: The preprocessed text.
    """

    # Convert text to lowercase
    text = raw_text.lower()

    # Tokenize the text into words
    words = word_tokenize(text)

    # Handling negations: If "not" or similar word is found, we merge it with the next word
    # Example: "not happy" becomes "not_happy"
    processed_words = []
    negations = ["not", "no", "never", "n't"]
    i = 0
    while i < len(words):
        if words[i] in negations and i+1 < len(words):
            processed_words.append(words[i] + '_' + words[i+1])
            i += 2  # Skip the next word
        else:
            processed_words.append(words[i])
            i += 1

    # Remove stopwords and any tokens that are just white spaces
    meaningful_words = [w for w in processed_words if w not in stops and w.strip()]

    # Rejoin words to form the final preprocessed text
    clean_text = " ".join(meaningful_words)

    return clean_text

  

# Function to predict emotion from a sentence
def predict_emotion(sentence, model_path):

    model_directory = "./saved_model"  # Adjust this path as necessary
    ensure_model_is_unzipped(model_directory)


    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    emotion_labels = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']

    
    # Preprocess the sentence
    preprocessed_sentence = preprocess_text(sentence)
    
    # Tokenize and prepare the inputs
    inputs = tokenizer(preprocessed_sentence, padding=True, truncation=True, max_length=512, return_tensors="pt")
    
    # Make a prediction
    # We ensure that no gradient is calculated as we are only making inference
    with torch.no_grad():
        predictions = model(**inputs).logits
    
    # Apply sigmoid and then threshold to convert logits to binary outputs
    threshold = 0.5
    sigmoid_predictions = torch.sigmoid(predictions).numpy()
    emotion_vector = [1 if pred > threshold else 0 for pred in sigmoid_predictions[0]]
    # Map the binary vector to the emotion names
    detected_emotions = [emotion_labels[i] for i, emotion in enumerate(emotion_vector) if emotion == 1]

    return emotion_vector, detected_emotions


def main(test_sentence):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.save_pretrained("./saved_model")
    print("Original text:", test_sentence)
    model_path = "./saved_model"
    emotion_vector, detected_emotions = predict_emotion(test_sentence, model_path)
    print("Emotion Vector:", emotion_vector)
    print("Detected Emotions:", detected_emotions)

if __name__ == "__main__":
    test_sentence = "I am so not happy to join it! This is not great :)"
    main(test_sentence)
    
