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
    scores = np.dot(w.T, data) + b
    predictions = (scores > 0).astype(int)  
    return (scores, predictions)

def SVM_lin(dataset: np.ndarray, K: float, C: float) -> tuple[np.ndarray, float]:
    '''
    Computes the w vector and b value for linear SVM 
    '''
    data, labels = dataset[:-1], dataset[-1]

    z = (2*labels - 1).reshape(1, labels.shape[0]) # zi = +-1, 1 if belongs to class Ht, -1 if belongs to class Hf
    
    hat_data = np.vstack([data, K*np.ones(data.shape[1])])

    hat_H = np.dot(z.T*hat_data.T, z*hat_data)

    alphas = np.ones((hat_data.shape[1], 1))/2

    boundaries = [(.0, C) for _ in alphas]

    def grad_hat_Ld(alphas, hat_H):
        return (np.dot(hat_H, alphas) - 1)

    def hat_Ld(alphas, hat_H):
        return  -(-1/2*np.dot(alphas.T, np.dot(hat_H, alphas)) + np.dot(alphas.T, np.ones(alphas.shape[0]))).item()

    best_alphas, _, _ = scipy.optimize.fmin_l_bfgs_b(hat_Ld, alphas*C, fprime=grad_hat_Ld, args=(hat_H,), bounds=boundaries, factr=0.0)
    w = (best_alphas*z*hat_data).sum(axis=1)
    return w[:-1], K*w[-1]

def SVM_lin_scores(evaluation_dataset: np.ndarray, w: np.ndarray, b: float) -> tuple[np.ndarray, np.ndarray]:
    '''
    Computes the scores for an evaluation dataset, given the model parameters.
    Returns a tuple with the scores and the predictions
    '''
    data = evaluation_dataset[:-1]
    scores = np.dot(w.T, data) + b
    predictions = (scores > 0).astype(int)
    return (scores, predictions)

def SVM_kernel(dataset, kernel=None):
    pass


