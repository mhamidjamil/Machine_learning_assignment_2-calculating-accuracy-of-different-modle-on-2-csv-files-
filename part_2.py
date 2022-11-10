# Muhammad Hamid Jamil
# SP19-BCS-098
import numpy as np
import pandas as pd
import time

from sklearn.model_selection import data_spliter
from sklearn.naive_bayes import gaussian
from sklearn.tree import Decision_tree_classifier
from sklearn.ensemble import random_forest

# loding data from file using pandas library
dataset = pd.read_csv("train.csv")

# ignoring infinit and null values
dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
dataset.fillna(999, inplace=True)

# managing labels and data
RequiredFields = ['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
dataset = dataset[RequiredFields]

# maintaining data in proper form so the code can understand
dataset['Sex'].replace({'male':1,'female':0}, inplace = True)
dataset['Embarked'].replace({'S':1,'C':0,'Q':2}, inplace = True)

X = dataset.drop(['Survived'], axis = 1)
Y = dataset['Survived']

# spiliting test and training data
X_train, X_test, Y_train, Y_test = data_spliter(X, Y, test_size = 0.2, random_state = 14)

# Training Model
classifier = gaussian()
classifier1 = Decision_tree_classifier(max_depth=3)
classifier2 = random_forest(n_estimators = 30, random_state = 15)

# Naive Bayes:

#Calculating Training Time
start = time.time()
classifier.fit(X_train, Y_train)
end = time.time()
trainingTimeNb = end - start

#Calculating Testing Time
start = time.time()
Prediction_Nb = classifier.predict(X_test)
end = time.time()
testingTimeNb = end - start

print('Efficiency of Naive Bayes: ')
print('Training Time: ', "{:.4f}".format(trainingTimeNb))
print('Testing Time: ', "{:.4f}".format(testingTimeNb))

# Calculating Accuracy of modle
Accuracy = classifier.score(X_test, Y_test)*100
print('Accuracy of Naive Bayes: ',"{:.2f}".format(Accuracy),'%')

# Desicion Tree of modle

# Calculating Training Time of modle
start = time.time()
classifier1.fit(X_train, Y_train)
end = time.time()
trainingTimeID3 = end - start

# Calculating Testing Time of modle
start = time.time()
Prediction_ID3 = classifier1.predict(X_test)
end = time.time()
testingTimeID3 = end - start

print('Efficiency of Decision Tree: ')
print('Training Time: ', "{:.4f}".format(trainingTimeID3))
print('Testing Time: ', "{:.4f}".format(testingTimeID3))

# Calculating Accuracy of modle
Accuracy = classifier1.score(X_test, Y_test)*100
print('Accuracy of Decision Tree: ',"{:.2f}".format(Accuracy),'%')

# Random Forest

# alculating Training Time of modle
start = time.time()
classifier2.fit(X_train, Y_train)
end = time.time()
trainingTimeRF = end - start

# Calculating Testing Time of modle
start = time.time()
Prediction_RF = classifier2.predict(X_test)
end = time.time()
testingTimeRF = end - start

print('Efficiency of Random Forest: ')
print('Training Time: ', "{:.4f}".format(trainingTimeRF))

print('Testing Time: ', "{:.4f}".format(testingTimeRF))

# calculating Accuracy of modle
Accuracy = classifier2.score(X_test, Y_test)*100
print('Accuracy of Random Forest: ',"{:.2f}".format(Accuracy),'%')
