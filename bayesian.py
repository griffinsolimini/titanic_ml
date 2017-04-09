import pandas as pd
import numpy as np
import setup

df = setup.preprocessed_training_set()

#  Temporary until I learn to bucketize stuff
df = df.drop(['Age', 'Fare'], 1)

N, features = df.shape

survived = [dict() for x in range(features-1)]
died = [dict() for x in range(features-1)]

columns = list(df)

for row in df.iterrows():
    row = row[1]

    pos = -1
    survived_flag = True
    for column in columns:
        val = row[column]
        if column == 'Survived':
            if val == 1:
                survived_flag = True
            else:
                survived_flag = False
        else:
            if survived_flag:
                if val in survived[pos].keys():
                    survived[pos][val] += 1
                else:
                    survived[pos][val] = 1
            else:
                if val in died[pos].keys():
                    died[pos][val] += 1
                else:
                    died[pos][val] = 1
        pos += 1

num_survived = 0
for key in survived[0].keys():
    num_survived += survived[0][key]

num_died = 0
for key in died[0].keys():
    num_died += died[0][key]

prob_survived = float(num_survived) / float(num_survived + num_died)
prob_died = float(num_died) / float(num_survived + num_died)

df = setup.preprocessed_test_set()

#  Temporary until I learn to bucketize stuff
df = df.drop(['Age', 'Fare'], 1)

columns = list(df)

classifications = []

for row in df.iterrows():
    row = row[1]

    pos = -1
    survived_flag = True
    
    prob_survived_est = prob_survived
    prob_died_est = prob_died

    for column in columns:
        val = row[column]
        
        if val in survived[pos].keys():
            prob_survived_est *= float(survived[pos][val]) / num_survived
        else:
            prob_survived_est *= 0

        if val in died[pos].keys():
            prob_died_est *= float(died[pos][val]) / num_died
        else:
            prob_died_est *= 0
            
        pos += 1

    if prob_survived_est > prob_died_est:
        classifications.append(1)
    else:
        classifications.append(0)
    
    setup.create_submission_csv(classifications)

