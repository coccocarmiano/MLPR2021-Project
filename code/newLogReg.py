import numpy as np
import utils
from os import listdir
from os.path import isfile, join
from classifiers import logreg, logreg_scores

def compute_scores(dataset, tag='', kfold=5, nPCA = [11, 9, 7, 5]):
    if tag != '':
        tag = tag + '_'
    
    nPCA = sorted(nPCA, reverse=True)
    if nPCA[0] > dataset.shape[0]-1:
        raise Exception('Invalid dimension in PCA')

    lambdas = np.logspace(-4, 3, 8)
    result = {}


    #Check if data has been already computed and try to load them
    files = [f for f in listdir('../trained') if (isfile(join('../trained', f)) and "logreg_" + tag + f"scores_" in f)]
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
        result[n] = np.empty((len(lambdas), 2, reduced_dataset.shape[1]))
        for i, l in enumerate(lambdas):
            print(f"Computing for n = {n}, l = {l}")
            folds_scores = []
            folds_labels = []

            for fold in folds:
                train_dataset = fold[0]
                test_dataset = fold[1]

                if tag == 'whiten_':
                    train_dataset, test_dataset = utils.normalize(train_dataset, other=test_dataset)
                    _, v = utils.whiten(train_dataset)
                    feats = train_dataset[:-1]
                    feats = v.T @ feats
                    train_dataset = np.vstack((feats, train_dataset[-1]))

                w, b = logreg(train_dataset, l)
                scores, _, _ = logreg_scores(test_dataset, w, b)
                folds_scores.append(scores)
                folds_labels.append(test_dataset[-1])

            result[n][i, 0, :] = np.concatenate(folds_scores)
            result[n][i, 1, :] = np.concatenate(folds_labels)
            acc = ((result[n][i, 0, :] > 0).astype(int) ==  result[n][i, 1, :]).sum() / len(result[n][i, 0, :])
            print(acc)

        with open("../trained/logreg_" + tag + f"scores_{kfold}K_{n}D.npy", 'wb') as fname:
            np.save(fname, result[n])
            
    return result, lambdas

if __name__ == '__main__':

    dataset = utils.load_train_data()
    result = compute_scores(dataset)

    norm_dataset = utils.normalize(dataset)
    result_norm = compute_scores(dataset, tag='norm')

    trd, trl = dataset[:-1, :], dataset[-1, :]
    trd = utils.gaussianize(trd)
    gau_dataset = np.vstack((trd, trl))
    result_gau = compute_scores(dataset, tag='gau')

    _, v = utils.whiten(norm_dataset)
    feats, labels = norm_dataset[:-1, :], norm_dataset[-1, :]
    feats = v.T @ feats
    whiten_dataset = np.vstack((feats, labels))

    result_whiten = compute_scores(whiten_dataset, tag='whiten')