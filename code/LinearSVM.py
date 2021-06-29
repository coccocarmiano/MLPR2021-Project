import numpy as np
import utils
import os
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
from classifiers import SVM_lin, SVM_lin_scores

def plot_PCA_lambda_minDCF(values):
    nPCA = list(values.keys())
    priors = values[nPCA[0]]

    brg = ['b', 'r', 'g']
    patches = [ptc.Patch(color=c, label=f"π = {p}") for p, c in zip(priors, brg)]

    fig, axs = plt.subplots(2, 2)
    fig.legend(handles=patches, handlelength=1, loc='center right', ncol=1, borderaxespad=0.1)

    for ax in axs.flat:
        ax.set(xlabel='λ', ylabel='minDCF', xscale='log')

    
    axs[0, 0].set_xlabel(None)
    axs[0, 1].set_xlabel(None)
    axs[0, 1].set_ylabel(None)
    axs[1, 1].set_ylabel(None)

    for n, ax in zip(nPCA, axs.flat):
        ax.set_title(f"PCA (n = {n})" if n < nPCA[0] else "No PCA")
        for i, p in enumerate(priors):
            lambdas, dcfs, _ = values[n][p]
            ax.minorticks_on()
            ax.plot(lambdas, dcfs, color=brg[i])


    fig.tight_layout()
    plt.subplots_adjust(right=0.825)
    fig.savefig('../img/' + 'logregquad_lambda_minDCF.jpg', format='jpg')

def compute_PCA_lambda_minDCF(dataset):
    priors = [.1, .5, .9]
    values_to_plot = {}
    biases = [0, 0.1, 5, 10]
    boundaries = [.1, 1]

    #Check if data has been already computed and try to load them
    data_computed = True
    values_to_plot = {}

    for i, p in enumerate(priors):
        if os.path.exists(f"../trained/linsvm_{i}.npy") is False:
            data_computed = False
            break
        else:
            loaded_data = np.load(f"../trained/linsvm_{i}.npy")
            nSplit = int(loaded_data.shape[0]/2)
            values_to_plot[p] = (biases, boundaries, loaded_data[0:nSplit], loaded_data[nSplit:])

    if data_computed is False:
        values_to_plot = {}
        _, folds = utils.kfold(dataset, n=5)
        tot_scores = np.empty((len(biases), len(boundaries), dataset.shape[1]))
        tot_label = np.empty((len(biases), len(boundaries), dataset.shape[1]))
        for i, k in enumerate(biases):
            for j, c in enumerate(boundaries):
                print(f"Computing for K = {k}, C = {c}")
                tmp_scores = []
                tmp_labels = []

                for fold in folds:
                    train_dataset = fold[0]
                    test_dataset = fold[1]

                    w, b = SVM_lin(train_dataset, k, c)
                    scores, _, _ = SVM_lin_scores(test_dataset, w, b)
                    tmp_scores.append(scores)
                    tmp_labels.append(test_dataset[-1])

                tot_scores[i, j, :] = np.concatenate(tmp_scores)
                tot_label[i, j, :] = np.concatenate(tmp_labels)
                acc = ((tot_scores[i, j] > 0).astype(int) == tot_label[i, j]).sum() / len(tot_scores[i, j])
                print(acc)

        for i, p in enumerate(priors):
            dcfs = np.empty((len(biases), len(boundaries)))
            thresholds = np.empty((len(biases), len(boundaries)))
            for j, _ in enumerate(biases):
                for k, _ in enumerate(boundaries):
                    dcfs[j, k], thresholds[j, k] = utils.min_DCF(tot_scores[j, k], tot_label[j, k], p)

            values_to_plot[p] = (biases, boundaries, dcfs, thresholds)
            with open(f"../trained/linsvm_{i}.npy", 'wb') as fname:
                np.save(fname, np.vstack([dcfs, thresholds]))
    
    return values_to_plot

if __name__ == '__main__':

    dataset = utils.load_train_data()
    values_to_plot = compute_PCA_lambda_minDCF(dataset)
    #plot_PCA_lambda_minDCF(values_to_plot)

    best_dcf = values_to_plot[0.5][2][0, 0]
    best_i = 0
    best_j = 0
    for i in range(len(values_to_plot[0.5][0])):
        for j in range(len(values_to_plot[0.5][1])):
            dcf = values_to_plot[0.5][2][i, j]
            if(dcf < best_dcf):
                best_i = i
                best_j = j
                best_dcf = dcf

    K = values_to_plot[0.5][0][best_i]
    C = values_to_plot[0.5][1][best_j]
    print(f"Best K, C: {K}, {C}")

    eval_dataset = utils.load_test_data()
    w, b = SVM_lin(dataset, K, C)
    scores, _, acc = SVM_lin_scores(eval_dataset, w, b)
    print(acc)