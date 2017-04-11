from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import pandas as pd
import numpy as np
import setup as s

train_X = s.preprocessed_training_set().drop(['Survived'], axis=1)
train_Y = s.preprocessed_training_set()['Survived']

logreg = LogisticRegression()
logreg.fit(train_X, train_Y)
print "Logistic Regression: " + str(logreg.score(train_X, train_Y))

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(train_X, train_Y)
print "Random Forest: " + str(random_forest.score(train_X, train_Y))

linsvm = LinearSVC()
linsvm.fit(train_X, train_Y)
print "Linear SVM: " + str(linsvm.score(train_X, train_Y))

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(train_X, train_Y)
print "KNearest: " + str(knn.score(train_X, train_Y))

nb = GaussianNB()
nb.fit(train_X, train_Y)
print "Naive Bayes: " + str(nb.score(train_X, train_Y))

test_X = s.preprocessed_test_set()

#  submission = pd.DataFrame({
        #  "PassengerId": pd.read_csv('data/test.csv')['PassengerId'],
        #  "Survived": classifier.predict(test_X)
    #  })
#  
#  submission.to_csv('data/result.csv', index=False)

