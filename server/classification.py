from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd

neigh = KNeighborsClassifier(n_neighbors=3)
supportVM = svm.SVC(probability=True)
forest = RandomForestClassifier(n_estimators = 200)
net = MLPClassifier(activation='tanh')

# 5
testCondition = [3,0,105,12,69,155,240,133,64,93,95,54,92,0,124,0,28,24,0,0,16,0,0,0,0,0,0,0,28,24,0,0,16,0,0,0,0,0,0,0,20,20,20,0,44,0,0,0,0,0,0,32,24,0,0,0,40,0,0,0,0,0,0,16,20,40,0,0,24,0,0,0,0,0,0,0,32,16,32,0,16,0,0,0,0,0,0,32,0,0,0,0,0,0,0,0,0,0,0,0,16,40,0,0,8,0,0,0,0,0,0,0,24,36,0,0,16,0,0,0,0,0,0,0,32,24,20,0,20,0,0,0,0,0,0,0,32,20,0,0,20,0,0,0,0,0,0,0,36,0,0,0,20,0,0,0,0,0,0,0.4,0.0,3.7,-3.8,0.0,0.0,1.0,0.0,0.6,0.6,0.1,0.0,3.7,-3.0,0.0,0.0,1.8,0.8,1.5,6.7,-0.3,0.0,1.3,-1.7,1.7,0.0,1.3,0.9,1.3,7.2,-0.5,-3.5,3.2,0.0,0.0,0.0,-1.4,-0.1,-1.8,-2.5,0.0,-0.4,2.1,-2.9,0.0,0.0,0.5,-0.3,-4.0,-5.0,0.2,0.0,2.8,-1.5,1.0,0.0,0.9,0.3,4.8,6.9,1.2,-4.9,0.0,0.0,0.0,0.0,-0.9,0.5,-7.8,-3.7,2.3,0.0,0.7,-11.4,0.0,0.0,0.2,2.1,-22.3,-5.1,3.3,0.0,5.6,-21.4,0.0,0.0,0.6,2.7,-31.8,-9.7,0.3,0.0,8.3,-11.6,1.0,0.0,1.1,0.4,0.3,1.6,-0.2,0.0,4.8,-4.3,0.0,0.0,1.4,0.1,3.3,3.8,-0.6,0.0,3.3,0.0,0.0,0.0,1.1,-0.1,5.9,5.4]

def classify():
    data = pd.read_csv('data/arrhythmia.csv')
    newData = pd.read_csv('uploads/data.csv')
    print(newData)
    labels = pd.read_csv('data/arrhythmiaNames.csv')
    Xtrain = data.iloc[:401,:279]
    ytrain = data.iloc[:401,279:]
    Xtest = data.iloc[401:451,:279]
    ytest = data.iloc[401:451,279:]
    # nearestNeighbours(Xtrain, ytrain,  Xtest, ytest)
    # supportVectorMachine(Xtrain, ytrain,  Xtest, ytest)
    # randomForest(Xtrain, ytrain,  Xtest, ytest)
    # neuralNet(Xtrain, ytrain, Xtest, ytest)

    eclf3 = VotingClassifier(estimators=[('nn', neigh), ('svm', supportVM), ('net', net), ('rf', forest)], voting='soft', weights=[3,1,1,1])

    eclf3.fit(Xtrain, ytrain)
    print(eclf3.predict(newData))
    return eclf3.predict(newData)

def nearestNeighbours(X, y, Xtest, ytest):
    neigh.fit(X, y)
    neigh.predict(testCondition)
    print(neigh.score(Xtest, ytest))

def supportVectorMachine(X, y, Xtest, ytest):
    supportVM.fit(X, y)
    supportVM.predict(testCondition)
    print(supportVM.score(Xtest, ytest))

def randomForest(X, y, Xtest, ytest):
    forest.fit(X, y)
    forest.predict(testCondition)
    print(forest.score(Xtest, ytest))

def neuralNet(X, y, Xtest, ytest):
    net.fit(X, y)
    net.predict(testCondition)
    print(net.score(Xtest, ytest))

classify()
