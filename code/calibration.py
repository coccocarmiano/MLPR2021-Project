import numpy as np
from classifiers import logreg

def calibrate_scores(scores, labels, p):
    data = np.vstack([scores, labels])
    alpha, beta = logreg(data)
    return alpha*scores + beta - np.log(p/(1-p))