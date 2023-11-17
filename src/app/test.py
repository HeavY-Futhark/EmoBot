from datasets import load_dataset
import pandas as pd
import numpy as np
import pickle
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt


dataset = load_dataset("sem_eval_2018_task_1")
data_train = dataset["train"]
data_test = dataset["test"]

x_train = data_train.iloc[:, 1:2]
y_train = data_train.iloc[:, 2:].to_numpy()
x_eval = data_test.iloc[:, 1:2]
y_eval = data_test.iloc[:, 2:].to_numpy()
train_df = pd.DataFrame([[x_train.iloc[k,0],y_train[k].tolist()] for k in range(0,int(y_train.shape[0]))], columns=["text", "target"])
eval_df = pd.DataFrame([[x_eval.iloc[k,0],y_eval[k].tolist()] for k in range(y_eval.shape[0])], columns=["text", "target"])
print("ok")