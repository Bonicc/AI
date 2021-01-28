import numpy as np

def MSE(hypothesis,label):
    return np.square(hypothesis-label)

def cross_entropy(hypothesis,label):
    return - ( label * np.log( hypothesis ) + (1-label) * np.log(1-hypothesis) )