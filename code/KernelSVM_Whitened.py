import utils
import classifiers
import numpy as np

filename = 'KernelSVM__Whitened.txt'

def get_kernel_function(gamma, regbias):
    def kernel_function(x1, x2):
        diff = x1 - x2
        diff = diff * diff
        diff = diff.sum()
        diff = diff * gamma
        diff = diff * -1
        value = np.exp(diff)
        value = value + regbias
        return value
    return kernel_function

if __name__ == '__main__':
    dataset = utils.load_train_data()
    _, folds = utils.kfold(dataset, n=5)
    outfile = open(filename, 'w')
    regbiases = [.01, .05, .1, .15, .25, .5]
    gammas = [.1, .15, .3, .5, .7, 1., 1.5, 3, 5, 10]
    bounds = [.1, .3, .5, .7, 1]

    for gamma in gammas:
        for regbias in regbiases:
            for bound in bounds:
                kernel_function = get_kernel_function(gamma, regbias)
                scores, labels = [], []
                for fold in folds:
                    train, test = fold[0], fold[1]
                    _, v = utils.whiten(train)
                    train, test = utils.normalize(train, other=test)
                    fold_labels = test[-1, :]
                    labels.append(fold_labels)

                    feats, temp = train[:-1, :], train[-1, :]
                    feats = v.T @ feats
                    train = np.vstack((feats, temp))

                    alphas = classifiers.DualSVM_Train(train, kernel_function, bound=bound)
                    train, alphas = utils.support_vectors(train, alphas)

                    feats, temp = test[:-1, :], test[-1, :]
                    feats = v.T @ feats
                    test = np.vstack((feats, temp))

                    fold_scores = classifiers.DualSVM_Score(train, kernel_function, alphas, test)
                    scores.append(fold_scores)
                
                scores = np.concatenate(scores)
                labels = np.concatenate(labels)
                mindcf, optimal_threshold = utils.minDCF(scores, labels, prior_t=.5)
                # Ignore the first field, is just handy for sorting
                print(f"{mindcf} |.| MinDCF: {mindcf:.4f}  -  Opt. Thr.: {optimal_threshold:.4f}  -  Gamma: {gamma:.2f}  -  Reg. Bias: {regbias:.2f}  -  C:   {bound:.2f}", file=outfile)

    outfile.close()
