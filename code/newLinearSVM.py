import numpy as np
import utils
from os import listdir
from os.path import isfile, join
from classifiers import SVM_lin, SVM_lin_scores

def compute_scores(dataset, tag='', kfold=5, nPCA = [11, 9, 7, 5]):
    if tag != '':
        tag = tag + '_'
    
    nPCA = sorted(nPCA, reverse=True)
    if nPCA[0] > dataset.shape[0]-1:
        raise Exception('Invalid dimension in PCA')

    biases = [0, 0.1, 1, 10, 100]
    boundaries = [.1, 1, 10]
    result = {}


    #Check if data has been already computed and try to load them
    files = [f for f in listdir('../trained') if (isfile(join('../trained', f)) and "linsvm_" + tag + f"scores_" in f)]
    PCAcomputed = []
    
    for f in files:
        n = int(f.split('_')[-1].split('D')[0])
        PCAcomputed.append(n)
        result[n] = np.load(join('../trained', f))

    PCAtoCompute = sorted([n for n in nPCA if n not in PCAcomputed], reverse=True)

    for n in PCAtoCompute:
        reduced_dataset = utils.reduce_dataset(dataset, n=n)
        _, folds = utils.kfold(reduced_dataset, n=kfold)
        # lambda, scores, labels
        result[n] = np.empty((len(biases), len(boundaries), 2, reduced_dataset.shape[1]))
        for i, k in enumerate(biases):
            for j, c in enumerate(boundaries):
                print(f"Computing for K = {k}, C = {c}")
                folds_scores = []
                folds_labels = []

                for fold in folds:
                    train_dataset = fold[0]
                    test_dataset = fold[1]

                    w, b = SVM_lin(train_dataset, k, c)
                    scores, _, _ = SVM_lin_scores(test_dataset, w, b)
                    folds_scores.append(scores)
                    folds_labels.append(test_dataset[-1])

                result[n][i, j, 0, :] = np.concatenate(folds_scores)
                result[n][i, j, 1, :] = np.concatenate(folds_labels)
                acc = ((result[n][i, j, 0, :] > 0).astype(int) ==  result[n][i, j, 1, :]).sum() / len(result[n][i, j, 0, :])
                print(acc)

        with open("../trained/linsvm_" + tag + f"scores_{kfold}K_{n}D.npy", 'wb') as fname:
            np.save(fname, result[n])
            
    return result, biases, boundaries

if __name__ == '__main__':

    dataset = utils.load_train_data()
    result = compute_scores(dataset)
    