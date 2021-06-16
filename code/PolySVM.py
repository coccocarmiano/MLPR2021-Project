import utils
import classifiers
import numpy as np

filename = 'PolySVM.txt'

def get_poly_function(c, d):
    def poly_function(x1, x2):
        value = x1.T @ x2
        value = value + c
        value = value ** d
        return value
    return poly_function

if __name__ == '__main__':
    dataset = utils.load_train_data()
    _, folds = utils.kfold(dataset, n=10)
    outfile = open(filename, 'w')
    constants = [0, .5, 1, 3, 5, 10]
    powers = [1, 2, 3, 4, 5]
    bounds = [.1, .3, .5, .7, 1]

    for power in powers:
        for constant in constants:
            for bound in bounds:
                poly_function = get_poly_function(power, constant)
                scores, labels = [], []
                for fold in folds:
                    train, test = fold[0], fold[1]
                    fold_labels = test[-1, :]
                    labels.append(fold_labels)

                    alphas = classifiers.DualSVM_Train(train, poly_function, bound=bound)
                    train, alphas = utils.support_vectors(train, alphas)
                    fold_scores = classifiers.DualSVM_Score(train, poly_function, alphas, test)
                    scores.append(fold_scores)
                
                scores = np.concatenate(scores)
                labels = np.concatenate(labels)
                mindcf, optimal_threshold = utils.minDCF(scores, labels, prior_t=.5)
                # Ignore the first field, is just handy for sorting
                print(f"{mindcf} |.| MinDCF: {mindcf:.4f}  -  Opt. Thr.: {optimal_threshold:.4f}  -  Power: {power:.2f}  -  Reg. Bias: {constant:.2f}  -  C:   {bound:.2f}", file=outfile)

    outfile.close()
