import numpy as np
import utils
from classifiers import DualSVM_Score, DualSVM_Train



def polysvm(folds, biases, degrees):

    def get_polyfunction(c, d):

        def f(x1, x2):
            return (x1.T @ x2 + c) ** d

        return f

    priors = np.linspace(.1, .9, 7)

    for degree in degrees:
        function = get_polyfunction(1, degree)
        for bias in biases:
            fscores, flabels = [], []
            for fold in folds:
                tel = fold[1][-1, :]

                alphas = DualSVM_Train(fold[0], function=function)
                scores = DualSVM_Score(fold[0], alphas, fold[1], bias=bias)
                fscores.append(scores)
                flabels.append(tel)
            
            fscores = np.concatenate(fscores)
            flabels = np.concatenate(flabels)
            
            mindcf, _, __ = utils.minDCF_SVM(fscores, flabels, priors)
            print(mindcf)


if __name__ == '__main__':
    dataset = utils.load_train_data()
    _, folds = utils.kfold(dataset)

    degrees = [4, 3, 2]
    biases = [1, 10, 100]
    polysvm(folds, biases, degrees)
