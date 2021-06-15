import numpy as np
import utils
from classifiers import DualSVM_Score, DualSVM_Train
from threading import Thread
import json

def polysvmt(folds, function, priors, bias, degree, c, norm):
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
        print(f"{mindcf} | {norm} | Effective Prior: {prior} | Optimal Threshold: {t} | Const: {c} | Power: {degree} | Bias: {bias}")


def kersvmt(folds, function, priors, bias, regbias, gamma, norm):
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
        print(f"{mindcf} | {norm} | Effective Prior: {prior} | Optimal Threshold: {t} | Gamma: {gamma} | RegBias: {regbias} : | Bias: {bias}")

def polysvm(folds, biases, consts, degrees, n):

    def get_polyfunction(c, d):

        def f(x1, x2):
            return (x1.T @ x2 + c) ** d

        return f

    priors = [.1, .5, .9]
    if n:
        n = "Norm"
    else:
        n = "Not norm"

    for c in consts:
        for degree in degrees:
            function = get_polyfunction(c, degree)
            for bias in biases:
                x = Thread(target=polysvmt, args=(folds, function, priors, bias, degree, c, n))
                x.start()


def kernelsvm(folds, gammas, regbiases, biases, n):

    priors = [.1, .5, .9]

    if n:
        n = "Norm"
    else:
        n = "Not Norm"

    def kernel(gamma, regbias):

        def f(x1, x2):
            diff = x1-x2
            value = (diff * diff).sum() * gamma * -1
            value = np.exp(value)
            return value + regbias

        return f

    priors = np.linspace(0.1, 0.9, 3)

    for regbias in regbiases:
        for gamma in gammas:
            function = kernel(gamma, regbias)
            for bias in biases:
                for fold in folds:
                    x = Thread(target=kersvmt, args=(folds, function, priors, bias, regbias, gamma, n))
                    x.start()


if __name__ == '__main__':
    dataset = utils.load_train_data()
    _, folds = utils.kfold(dataset, 20)
    

    degrees = [2, 3, 4]
    biases = [1, 5, 10, 20]
    regbiases = [.01, .05, .1, .15, .20]
    gammas = [.05, .1, .15, .2, .3, .5, .7, .9, 1]
    consts = [0, 1, 3, 5, 10]

    print("polysvm:")
    polysvm(folds, biases, consts, degrees, False)
    print("kernelsvm:")
    kernelsvm(folds, gammas, regbiases, biases, False)

    feats, labels = dataset[:-1, :], dataset[-1, :]
    feats = utils.gaussianize(feats)
    dataset = np.vstack((feats, labels))
    _, folds = utils.kfold(dataset, 5)

    print("polysvm (G):")
    polysvm(folds, biases, consts, degrees, True)
    print("kernelsvm (G):")
    kernelsvm(folds, gammas, regbiases, biases, True)
