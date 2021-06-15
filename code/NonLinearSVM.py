import numpy as np
import utils
from classifiers import DualSVM_Score, DualSVM_Train
import json

f = None
obj = {}

def polysvm(folds, biases, degrees):

    def get_polyfunction(c, d):

        def f(x1, x2):
            return (x1.T @ x2 + c) ** d

        return f

    priors = [.1, .5, .9]

    for degree in degrees:
        function = get_polyfunction(1, degree)
        for bias in biases:
            fscores, flabels = [], []
            for fold in folds:
                tel = fold[1][-1, :]
                alphas = DualSVM_Train(fold[0], function=function, factr=1., bound=1.)
                scores = DualSVM_Score(fold[0], alphas, fold[1], bias=bias)

                fscores.append(scores)
                flabels.append(tel)
            
            scores = np.concatenate(fscores)
            labels = np.concatenate(flabels)
            
            for prior in priors:
                mindcf, t = utils.minDCF(scores, labels, prior_t=prior, thresholds=scores)
                print(f"{mindcf} | Effective Prior: {prior} | Optimal Threshold: {t} | Power: {degree} | Bias: {bias}")


def kernelsvm(folds, gammas, biases):

    priors = [.1, .5, .9]

    def kernel(gamma):

        def f(x1, x2):
            diff = x1-x2
            value = (diff * diff).sum() * gamma * -1
            value = np.exp(value)
            return value + bias

        return f

    priors = np.linspace(0.1, 0.9, 3)

    for gamma in gammas:
        function = kernel(gamma)
        for bias in biases:
            fscores, flabels = [], []
            for fold in folds:
                tel = fold[1][-1, :]

                alphas = DualSVM_Train(fold[0], function=function, factr=1)
                scores = DualSVM_Score(fold[0], alphas, fold[1], bias=bias)
                fscores.append(scores)
                flabels.append(tel)
            
            scores = np.concatenate(fscores)
            labels = np.concatenate(flabels)
            
            for prior in priors:
                mindcf, t = utils.minDCF(scores, labels, prior_t=prior, thresholds=scores)
                print(f"{mindcf} | Effective Prior: {prior} | Optimal Threshold: {t} | Gamma: {gamma} | Bias: {bias}")


if __name__ == '__main__':
    dataset = utils.load_train_data()
    _, folds = utils.kfold(dataset, 4)
    

    degrees = [2, 3, 4]
    biases = [1, 10, 100]
    gammas = [.1, .3, .5, .7, 1]

    print("polysvm:")
    polysvm(folds, biases, degrees)
    print("kernelsvm:")
    kernelsvm(folds, gammas, biases)

    feats, labels = dataset[:-1, :], dataset[-1, :]
    feats = utils.normalize(feats)
    dataset = np.vstack((feats, labels))
    _, folds = utils.kfold(dataset, 4)

    #print("polysvm (n):")
    #polysvm(folds, biases, degrees)
    #print("kernelsvm (n):")
    #kernelsvm(folds, gammas, biases)
