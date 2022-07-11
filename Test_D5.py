import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from Data_Science_Toolkit import ClassifierAlgorithm, simplekNNClassifier, Experiment, DecisionTree, Tree, node, mergesort,kdTreeKNNClassifier

#download and organize dataset
df = pd.read_csv("iris.csv")
df = df.sample(frac=1)
features = df.drop(['variety'],axis=1)
features=np.array(features)

labels = df['variety']
labels=np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33)

print("\n### " + "Tests for functionalities and attributes of kdTreeKNNClassifier subclass"+ " ###\n")
print("*Test constructor:")
c = kdTreeKNNClassifier()
print("\nTest train function:")
c.train(X_train,y_train)
print("\nTest test function:")
print(c.test(X_test,y_test))

print("\n### " + "Tests for functionalities and attributes of Experiment class"+ " ###\n")
classifiers = [c]
print("*Test constructor:")
ep = Experiment(features,labels,classifiers)
print("\nTest runCrossVal function:")
print(ep.runCrossVal())
print("\nTest score function:")
print(ep.score())
print("\nTest _confusionMatrix function:")
ep._confusionMatrix()


