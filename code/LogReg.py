import numpy as np
import utils
import os
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
from classifiers import logreg, logreg_scores

def latex(toprint):
    outfiletex = '../data/logreg_acc.tex'
    f = open(outfiletex, "w")
    print(r"\caption{Logistic Regression}\label{tab:logreg}", file=f)
    print(r"\begin{center}", file=f)
    print(r"\begin{tabular}{|c|c||c|c|}", file=f)
    print(r"\hline", file=f)
    print(r"\ & PCA & Error Rate & $DCF$\\", file=f)
    print(r"\hline", file=f)
    
    toprint.sort(key=lambda x: min(x[0]))
    toprint = toprint[:3]
    for tup in toprint:
        for i in range(len(tup[0])):
            print(f"$\\pi_T = {tup[3][i]:.2f}$ & {tup[1]} & {tup[2][i]*100:.2f} & {tup[0][i]:.3f} \\\\", file=f)
        print(r"\hline", file=f)

    print(r"\end{tabular}", file=f)
    print(r"\end{center}", file=f)

    f.close()

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
    fig.savefig('../img/' + 'logreg_lambda_minDCF.jpg', format='jpg')


def compute_PCA_lambda_minDCF(dataset):
    lambdas = np.logspace(-4, 3, 8)
    priors = [.1, .5, .9]
    nPCA = [11, 9, 7, 5]
    dcfs = {i: np.empty(len(lambdas)) for i in range(len(priors))}
    thresholds = {i: np.empty(len(lambdas)) for i in range(len(priors))}
    values_to_plot = {}
    
    #Check if data has been already computed and try to load them
    data_computed = True
    values_to_plot = {}
    for n in nPCA:
        values_to_plot[n] = {}
        for i, p in enumerate(priors):
            if os.path.exists(f"../trained/logreg_dcfs_{n}_{i}.npy") is False:
                data_computed = False
                break
            else:
                loaded_data = np.load(f"../trained/logreg_dcfs_{n}_{i}.npy")
                values_to_plot[n][p] = (lambdas, loaded_data[0], loaded_data[1])

    if data_computed is False:
        #values_to_plot = {}
        #for n in nPCA:
        #    reduced_dataset = utils.reduce_dataset(dataset, n=n)
        #    _, folds = utils.kfold(reduced_dataset, n=5)
        #    values_to_plot[n] = {}
        #    for i, p in enumerate(priors):
        #        for j, l in enumerate(lambdas):
        #            print(f"Computing for n = {n}, p = {p}, l = {l}")
        #            tot_scores = []
        #            tot_label = []
        #            for fold in folds:
        #                trdata = fold[0]
        #                tedata = fold[1]
        #                w, b = logreg(trdata, l)
        #                scores, _, _ = logreg_scores(tedata, w, b)
        #                tot_scores.append(scores)
        #                tot_label.append(tedata[-1])
        #            tot_scores = np.concatenate(tot_scores)
        #            tot_label = np.concatenate(tot_label)
        #            dcfs[j], thresholds[j] = utils.min_DCF(tot_scores, tot_label, p)
        #        with open(f"../trained/logreg_dcfs_{n}_{i}.npy", 'wb') as fname:
        #            np.save(fname, np.vstack([dcfs, thresholds]))
        #    
        #    values_to_plot[n][p] = (lambdas, dcfs, thresholds)
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
                    train_dataset = fold[0]
                    test_dataset = fold[1]

                    w, b = logreg(train_dataset, l)
                    scores, _, _ = logreg_scores(test_dataset, w, b)
                    tot_scores[i].append(scores)
                    tot_label[i].append(test_dataset[-1])

                tot_scores[i] = np.concatenate(tot_scores[i])
                tot_label[i] = np.concatenate(tot_label[i])
                acc = ((tot_scores[i] > 0).astype(int) == tot_label[i]).sum() / len(tot_scores[i])
                print(acc)

            for i, p in enumerate(priors):
                for j in range(len(lambdas)):
                    dcfs[i][j], thresholds[i][j] = utils.min_DCF(tot_scores[j], tot_label[j], p)

                values_to_plot[n][p] = (lambdas, dcfs[i], thresholds[i])
                with open(f"../trained/logreg_dcfs_{n}_{i}.npy", 'wb') as fname:
                    np.save(fname, np.vstack([dcfs, thresholds]))
                
    return values_to_plot

if __name__ == '__main__':

    dataset = utils.load_train_data()
    values_to_plot = compute_PCA_lambda_minDCF(dataset)
    plot_PCA_lambda_minDCF(values_to_plot)
