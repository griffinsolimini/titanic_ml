from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import setup as s

train_X = s.preprocessed_training_set().drop(['Survived'], axis=1)
train_Y = s.preprocessed_training_set()['Survived']

#  logreg = LogisticRegression()
#  logreg.fit(train_X, train_Y)
#  print "Logistic Regression: " + str(logreg.score(train_X, train_Y))
#  
rf = RandomForestClassifier(n_estimators=1000, max_depth=3)
rf.fit(train_X, train_Y)
print "Random Forest: " + str(rf.score(train_X, train_Y))
#  

#  err_values = []
#  for i in range(1, 140):
    #  random_forest = RandomForestClassifier(n_estimators=i)
    #  random_forest.fit(train_X, train_Y)
    #  err_values.append(random_forest.score(train_X, train_Y))
#  
#  plt.plot(range(1, 140), err_values, '-')
#  
#  plt.ylabel('training error')
#  plt.xlabel('number of estimators')
#  plt.show()

mlp = MLPClassifier()
mlp.fit(train_X, train_Y)
print "Neural Net: " + str(mlp.score(train_X, train_Y))

#  linsvm = LinearSVC()
#  linsvm.fit(train_X, train_Y)
#  print "Linear SVM: " + str(linsvm.score(train_X, train_Y))

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(train_X, train_Y)
print "KNN: " + str(knn.score(train_X, train_Y))
#  
#  err_values = []
#  for i in range(1, 100):
    #  knn = KNeighborsClassifier(n_neighbors = i)
    #  knn.fit(train_X, train_Y)
    #  err_values.append(knn.score(train_X, train_Y))
#  
#  plt.plot(range(1,100), err_values, '-')
#  
#  plt.ylabel('training error')
#  plt.xlabel('number of neighbors')
#  plt.show()

#  nb = GaussianNB()
#  nb.fit(train_X, train_Y)
#  print "Naive Bayes: " + str(nb.score(train_X, train_Y))

test_X = s.preprocessed_test_set()

knn_submission = pd.DataFrame({ 
    "PassengerId": pd.read_csv('data/test.csv')['PassengerId'],
    "Survived": knn.predict(test_X)
})
  
knn_submission.to_csv('data/knn_result.csv', index=False)

rf_submission = pd.DataFrame({ 
    "PassengerId": pd.read_csv('data/test.csv')['PassengerId'],
    "Survived": rf.predict(test_X)
})
  
rf_submission.to_csv('data/rf_result.csv', index=False)

