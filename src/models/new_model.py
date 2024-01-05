# train_model.py : current version using bert-base-uncased transformers

"""
train_model.py: Script to fine-tune a transformer model on a multi-label classification task.
"""

import numpy as np
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.metrics import classification_report, accuracy_score,  hamming_loss
import torch
#export WANDB_DISABLED=true #write this in your terminal to not get wand message, otherwise type 3

def main():

    def compute_metrics(pred):

        """
        Calculate evaluation metrics for the model prediction.

        Args:
            pred: `EvalPrediction` object. This object must have the following attributes:
                - `predictions`: Predictions of the model.
                - `label_ids`: Ground truth labels.

        Returns:
            A dictionary with calculated metrics including accuracy, hamming loss, and classification report.
        """
        #Defining multi-label classification problem using binary classes
        preds = pred.predictions
        threshold = 0.5
        preds_binary = np.where(preds > threshold, 1, 0)

        #Metrics to calculate
        accuracy = accuracy_score(pred.label_ids, preds_binary)
        hamming = hamming_loss(pred.label_ids, preds_binary)
        report = classification_report(pred.label_ids, preds_binary, target_names=emotion_columns, zero_division=0)

        # Save the classification report to a file
        with open('classification_report.txt', 'w') as output_file:
            print(report, file=output_file)

        
        return {
            'accuracy': accuracy,
            'hamming_loss': hamming,
            'classification_report': report  
        }

    # Collecting dataset from hugginface and defining the columns names
    dataset = load_dataset('sem_eval_2018_task_1', 'subtask5.english')
    
    emotion_columns = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']
    print(dataset['train'][0])
    # Pretrained model and tokenizer 
    model_name = "bert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(emotion_columns), problem_type="multi_label_classification")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess_function(batch):

        """
        Preprocess the data batch for input into the model.

        Args:
            batch: A dictionary representing a batch of data points.

        Returns:
            A dictionary mapping model inputs to their respective tensors.
        """

        # Tokenize the text to input_ids, attention_mask
        tokenized_inputs = tokenizer(batch['Tweet'], padding='max_length', truncation=True, max_length=512)

        # Prepare a list for labels
        labels = []
        # For each example in the batch, read its emotion data and convert it to 0s and 1s
        for i in range(len(batch['Tweet'])):  # Use a counter to access elements
            label = [float(batch[emotion][i]) for emotion in emotion_columns]  # Convert True/False to 1/0
            labels.append(label)

        # Return the tokenized inputs and labels as the output of the preprocess function
        return {
            'input_ids': tokenized_inputs['input_ids'],
            'attention_mask': tokenized_inputs['attention_mask'],
            'labels': labels
        }


    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    #Training parameters
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        evaluation_strategy="epoch",
        report_to=[]  #disable wandb
    )

    # Initialize the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        compute_metrics=compute_metrics  # Calculate metrics during evaluation

    )

    #Train the model
    trainer.train()
    #Evaluate the model
    results = trainer.evaluate()
    print(results)
    #Save the trained model
    model.save_pretrained("./saved_model")
    torch.save(model.state_dict(), "./saved_model/pytorch_model.bin")



#Run the main function
if __name__ == "__main__":
    main()


