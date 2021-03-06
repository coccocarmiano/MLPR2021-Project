import utils
import classifiers
import numpy as np

filename = '../data/KernelSVM.txt'

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
    _, folds = utils.kfold(dataset, n=3)
    outfile = open(filename, 'w')
    regbiases = [.005, .01, .05, .1]
    gammas = [.05, .01, .1, .5]
    bounds = [.5, .1, 1.5]
    npca = [11, 10, 9, 8]
    w, v = utils.PCA(dataset)

    for gamma in gammas:
        for regbias in regbiases:
            for n in npca:
                vt = v[:, :n]
                for bound in bounds:
                    kernel_function = get_kernel_function(gamma, regbias)
                    scores, labels = [], []
                    for fold in folds:
                        train, test = fold[0], fold[1]
                        train, test = np.vstack((vt.T @ train[:-1, :], train[-1])), np.vstack((vt.T @ test[:-1, :], test[-1]))
                        fold_labels = test[-1, :]
                        labels.append(fold_labels)

                        alphas = classifiers.DualSVM_Train(train, kernel_function, bound=bound)
                        train, alphas = utils.support_vectors(train, alphas)
                        fold_scores = classifiers.DualSVM_Score(train, kernel_function, alphas, test)
                        scores.append(fold_scores)

                    scores = np.concatenate(scores)
                    labels = np.concatenate(labels)
                    mindcf, optimal_threshold = utils.minDCF(scores, labels, prior_t=.5)
                    # Ignore the first field, is just handy for sorting
                    print(f"{mindcf} |.| MinDCF: {mindcf:.4f}  -  PCA: {n} - Opt. Thr.: {optimal_threshold:.4f}  -  Gamma: {gamma:.4f}  -  Reg. Bias: {regbias:.4f}  -  C:   {bound:.4f}", file=outfile)
                    np.save(f'../data/KernelSVM-PCA{n}-RegBias{regbias}-Gamma{gamma}-Bound{bound}Scores.npy', scores)

    outfile.close()
