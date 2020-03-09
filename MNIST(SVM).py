#import required libraries
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import math, time 
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
#read train and test files
#train_file = pd.read_csv('mnist_train.csv')
#test_file = pd.read_csv('mnist_test.csv')

#first few rows of the test and train files
#train_file.head()
#test_file.head()
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


#list of all digits that are going to be predicted
np.sort(train_file.label.unique())

#define the number of samples for training set and for validation set from the training set
num_train,num_validation = int(len(train_file)*0.8),int(len(train_file)*0.2)

#calculating the number of training and validation sets
num_train,num_validation

#generate training data from train_file
x_train,y_train=train_file.iloc[:num_train,1:].values,train_file.iloc[:num_train,0].values

#generate validationn data from train_file
x_validation,y_validation=train_file.iloc[num_train:,1:].values,train_file.iloc[num_train:,0].values

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)
sc_test = sc_X.transform(x_test)

from sklearn.svm import SVC
print('SVM Classifier with gamma = 0.1; Kernel = Polynomial')
classifier = SVC(gamma=0.1, kernel='poly', random_state = 0)
classifier.fit(x_train,y_train)

import seaborn as sns

sns.countplot(y_train)
#num_test = int(len(test_file)*1.0)
#x_test,y_test=test_file.iloc[:num_test,1:].values,test_file.iloc[:num_test,0].values

#to predict the labels in the test set
y_pred = classifier.predict(x_test)

#accuracy 
from sklearn.metrics import accuracy_score,confusion_matrix
model_acc = classifier.score(x_test, y_test)
test_acc = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test,y_pred)



