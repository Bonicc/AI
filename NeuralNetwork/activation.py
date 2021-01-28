import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    x = np.array(x)
    return np.array((x > 0) * x)
