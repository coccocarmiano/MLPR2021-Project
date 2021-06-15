import numpy as np
import utils
import os
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
from classifiers import logreg, logreg_scores

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

def expand_feature_space(dataset):
    data = dataset[:-1]
    def vecxxT(x):
        x = x[:, None]
        xxT = x.dot(x.T).reshape(x.size**2, order='F')
        return xxT
    expanded = np.apply_along_axis(vecxxT, 0, data)
    return np.vstack([expanded, dataset])

def compute_PCA_lambda_minDCF(dataset):
    lambdas = np.logspace(-5, -3, 3)
    priors = [.1, .5, .9]
    nPCA = [11, 9, 7, 5]
    values_to_plot = {}
    
    #Check if data has been already computed and try to load them
    data_computed = True
    values_to_plot = {}
    for n in nPCA:
        if data_computed is False:
            break

        values_to_plot[n] = {}
        for i, p in enumerate(priors):
            if os.path.exists(f"../trained/logregquad_dcfs_{n}_{i}.npy") is False:
                data_computed = False
                break
            else:
                loaded_data = np.load(f"../trained/logregquad_dcfs_{n}_{i}.npy")
                values_to_plot[n][p] = (lambdas, loaded_data[0], loaded_data[1])

    if data_computed is False:
        values_to_plot = {}
        for n in nPCA:
            reduced_dataset = utils.reduce_dataset(dataset, n=n)
            _, folds = utils.kfold(reduced_dataset, n=5)
            values_to_plot[n] = {}
            tot_scores = {}
            tot_label = {}
            for i, l in enumerate(lambdas):
                print(f"Computing for n = {n}, l = {l}")
                tot_scores[i] = []
                tot_label[i] = []

                for fold in folds:
                    train_dataset = expand_feature_space(fold[0])
                    test_dataset = expand_feature_space(fold[1])

                    w, b = logreg(train_dataset, l)
                    scores, _, _ = logreg_scores(test_dataset, w, b)
                    tot_scores[i].append(scores)
                    tot_label[i].append(test_dataset[-1])

                tot_scores[i] = np.concatenate(tot_scores[i])
                tot_label[i] = np.concatenate(tot_label[i])
                acc = ((tot_scores[i] > 0).astype(int) == tot_label[i]).sum() / len(tot_scores[i])
                print(acc)

            for i, p in enumerate(priors):
                dcfs = np.empty(len(lambdas))
                thresholds = np.empty(len(lambdas))
                for j in range(len(lambdas)):
                    dcfs[j], thresholds[j] = utils.min_DCF(tot_scores[j], tot_label[j], p)

                values_to_plot[n][p] = (lambdas, dcfs, thresholds)
                with open(f"../trained/logregquad_dcfs_{n}_{i}.npy", 'wb') as fname:
                    np.save(fname, np.vstack([dcfs, thresholds]))
    
    return values_to_plot

if __name__ == '__main__':

    dataset = utils.load_train_data()
    values_to_plot = compute_PCA_lambda_minDCF(dataset)
    plot_PCA_lambda_minDCF(values_to_plot)

    dim = 11
    best_dcf = values_to_plot[dim][0.5][1][0]
    best_index = 0
    for index, dcf in enumerate(values_to_plot[dim][0.5][1]):
        if(dcf < best_dcf):
            best_index = index

    l = values_to_plot[dim][0.5][0][best_index]
    t = values_to_plot[dim][0.5][2][best_index]
    print(f"Best lambda: {l}")

    eval_dataset = utils.load_test_data()
    eval_dataset = expand_feature_space(eval_dataset)
    dataset = expand_feature_space(dataset)
    w, b = logreg(dataset, l, precision=True)
    scores, _, acc = logreg_scores(eval_dataset, w, b, t=t)
    print(acc)