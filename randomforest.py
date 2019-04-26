import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import setup

df = setup.preprocessed_training_set()

#Get samples and labels from training set, and samples form test set
Labels = df['Survived']

Samples = df.drop(['Survived'],1)

test = setup.preprocessed_test_set()

#Create and train random forest classifier
clf = RandomForestClassifier(n_estimators=15,criterion='gini')

'''
#Try simple decision tree
clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0)

#Try Extremely Randomized Trees
clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
'''


clf = clf.fit(Samples,Labels)

#Test the classifier
res = clf.predict(test)

setup.create_submission_csv(res)

#Scores with certain parameters
#Your submission scored 0.76555. RandomForest(n_estimators=15)
#Your submission scored 0.75120. RandomForest(n_estimators=1)      (decision tree)
#Your submission scored 0.75120. RandomForest(n_estimators=15,criterion='entropy')
#Your submission scored 0.76555. DecisionTree
#Your submission scored 0.73684. ExtraTrees(n_estimator=10)





'''
N, features = df.shape
Labels_Train = Labels[:801]
Samples_Train = Samples[:801]
Labels_Test = Labels[801:891]
Samples_Test = Samples[801:891]
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(Samples_Train,Labels_Train)
res = clf.predict(Samples_Test)
diff = res - Labels_Test
np.count_nonzero(diff)
'''