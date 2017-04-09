import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import setup

df = setup.preprocessed_training_set()

df['Bias'] = np.ones(len(df['Survived']))

# Build x matrix and y vector
Y = np.matrix(df['Survived'].as_matrix()).T
X = np.matrix(df.drop(['Survived'], 1).as_matrix())
N, features = X.shape

def loss(w):
    return float(np.log(1 + np.exp(-w.T * X.T * Y)) / N)

def gradient(w, eta):
    h = float(np.exp(-w.T * X.T * Y))
    return w - eta * (1.0 / N) * h / (1 + h) * (-X.T * Y) 

def train_increment(w, eta):
    return gradient(w, eta)

def train(eta, T):
    w = np.zeros((features,1))
    for t in range(0, T):
        w = train_increment(w, eta) 
    return w

def run_experiment(eta_values):
    T = 1000
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

    best_T = 0 
    lowest_err = float("inf")

    max_T = 1000

    T_values = range(1, max_T + 1)
    err_values = [] 

    w = np.zeros((features,1))

    for T in T_values:
        w = train_increment(w, best_eta)
        
        err = loss(w)
        
        err_values.append(err)

        if err < lowest_err:
            best_T = T
            lowest_err = err


    w = train(best_eta, best_T)
    
    print 1 / (1 + np.exp(w.T * X.T))
    #  print w.T * X.T
     
    #  print "best T: " + str(best_T)
    #  print "best eta: " + str(best_eta)

    #  plt.plot(T_values, err_values, '-')
#  
    #  plt.title('Validation Error vs. T Values')
    #  plt.xlabel('T Value')
    #  plt.ylabel('Validation Error')
#  
    #  plt.show()

if __name__ == '__main__':
    eta_values = [5e-7]
    run_experiment(eta_values)

