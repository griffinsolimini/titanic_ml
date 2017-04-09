import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import setup
import math

df = setup.preprocessed_training_set()

df['Bias'] = np.ones(len(df['Survived']))

# Build x matrix and y vector
Y = np.matrix(df['Survived'].as_matrix()).T
X = np.matrix(df.drop(['Survived'], 1).as_matrix())
N, features = X.shape

def loss(w):
    return float((X * w - Y).T * (X * w - Y) / N)

def gradient(w, eta):
    return w - eta * (2.0 / N) * ((X * w - Y).T * X).T

def train_increment(w, eta):
    return gradient(w, eta)

def train(eta, T):
    w = np.zeros((features,1))
    for t in range(0, T):
        w = train_increment(w, eta) 
    return w

def run_experiment(eta_values):
    # train classifier
    T = 20000
    best_eta = 0 
    lowest_err = float("inf")

    for eta in eta_values:
        w = train(eta, T)
        
        err = loss(w)

        print "eta: " + str(eta)
        print "training error: " + str(err)
        print
        
        if err < lowest_err:
            best_eta = eta
            lowest_err = err

    # classify test set
    df = setup.preprocessed_test_set()
    df['Bias'] = np.ones(len(df['Fare']))
    X = np.matrix(df.as_matrix())
    
    classifications = [] 
    for value in (w.T * X.T).A1:
        if value >= .5:
            classifications.append(1)
        else:
            classifications.append(0)

    setup.create_submission_csv(classifications)

if __name__ == '__main__':
    eta_values = [5e-6]
    run_experiment(eta_values)

