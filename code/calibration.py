import numpy as np
from classifiers import logreg

def calibrate_scores(trscores, trlabels, scores, p):
    """
        trscores: data to train the model
        trlabels: labels to train the model
        scores: scores to calibrate
        p: prior
    """
    data = np.vstack([trscores, trlabels])
    alpha, beta = logreg(data)
    return alpha*scores + beta - np.log(p/(1-p))