import numpy as np
import scipy
from utils import mcol

def logreg(dataset: np.ndarray, l: float=10**-3) -> tuple[np.ndarray, float]:
    '''
    Computes the w vector and b value for the logistic regression
    '''
    data, labels = dataset[:-1], dataset[-1]

    def logreg_obj(v):
        w, b = v[:-1], v[-1]
        w = mcol(w)
        # computes objective function
        partial = 0
        for i in range(data.shape[1]):
            partial = partial + labels[i]*np.log1p(np.exp(- np.dot(w.T, data[:, i]) - b)) + (1 - labels[i])*np.log1p(np.exp(np.dot(w.T, data[:, i]) + b))
        partial =  partial/data.shape[1] + l/2*np.dot(w.T, w).flatten()

        return partial

    v0 = np.zeros(data.shape[0] + 1)
    v, _, _ = scipy.optimize.fmin_l_bfgs_b(logreg_obj, v0, approx_grad=True, factr=0)
    return v[:-1], v[-1]

def logreg_scores(evaluation_dataset: np.ndarray, w: np.ndarray, b: float) -> tuple[np.ndarray, np.ndarray]:
    '''
    Computes the scores for an evaluation dataset, given the model parameters.
    Returns a tuple with the scores and the predictions
    '''
    data = evaluation_dataset[:-1]
    scores = np.array([np.dot(w.T, x)+b for x in data.T])
    predictions = (scores > 0).astype(int)  
    return (scores, predictions)

def SVM_lin():
    pass

def SVM_kernel(dataset, kernel=None):
    pass


