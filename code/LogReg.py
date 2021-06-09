import numpy as np
import utils
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


def plot_PCA_lambda_minDCF(dataset, tag=None):
    r, c = dataset[:-1].shape
    lambdas = np.logspace(-9, 3, 13)
    dcfs = np.empty(len(lambdas))
    priors = [.1, .5, .9]
    brg = ['b', 'r', 'g']
    patches = [ptc.Patch(color=c, label=f"π = {p}") for p, c in zip(priors, brg)]
    nPCA = [11, 9, 7, 5]
    fig, axs = plt.subplots(2, 2)
    fig.legend(handles=patches, handlelength=1, loc='upper right')
    plots = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    for ax in axs.flat:
        ax.set(xlabel='λ', ylabel='minDCF', xscale='log')
    
    axs[0, 0].set_xlabel(None)
    axs[0, 1].set_xlabel(None)
    axs[0, 1].set_ylabel(None)
    axs[1, 1].set_ylabel(None)

    for n, ax in zip(nPCA, axs.flat):
        reduced_dataset = utils.reduce_dataset(dataset, n=n)
        _, folds = utils.kfold(reduced_dataset, n=5)
        ax.set_title(f", with PCA (n = {n})" if n < r else "No PCA")
        for i, p in enumerate(priors):
            for j, l in enumerate(lambdas):
                print(f"Computing for n = {n}, p = {p}, l = {l}")
                tot_scores = []
                tot_label = []
                for fold in folds:
                    trdata = fold[0]
                    tedata = fold[1]
                    w, b = logreg(trdata, l)
                    scores, _, _ = logreg_scores(tedata, w, b)
                    tot_scores.append(scores)
                    tot_label.append(tedata[-1])
                tot_scores = np.concatenate(tot_scores)
                tot_label = np.concatenate(tot_label)
                dcfs[j] = utils.min_DCF(tot_scores, tot_label, p)
            ax.plot(lambdas, dcfs, color=brg[i])

    fig.tight_layout()
    plt.show(block=False)
    fig.savefig('../img/' + 'logreg_lambda_minDCF.jpg', format='jpg')


if __name__ == '__main__':

    dataset = utils.load_train_data()
    plot_PCA_lambda_minDCF(dataset)

    #def tmp():
    #    trdataset = utils.load_train_data()
    #    tedataset = utils.load_test_data()
    #    _, folds = utils.kfold(trdataset, n=5)
    #    toprint = []
    #    lambdas = [0, 10**-9, 10**-6, 10**-3, 0.1]
    #    priors = [.1, .50, .9]
    #    tot_scores = []
    #    tot_labels = []
    #    for fold in folds:
    #        trdata = fold[0]
    #        tedata = fold[1]
    #        telabel = fold[1][-1, :]
    #        for l in lambdas:
    #            print(f"Computing for lambda: {l}")
    #            w, b = logreg(trdata, l)
    #            scores, _, _ = logreg_scores(tedata, w, b)
    #            tot_scores.append(scores)
    #            tot_labels.append(telabel)
    #    scores = np.concatenate(tot_scores)
    #    labels = np.concatenate(tot_labels)
    #    
    #    dcfs, ers = [], []
    #    for prior in priors:
    #        t = - np.log( prior / (1-prior))
    #        pscores = scores > t
    #        er = (pscores != labels).sum() / len(labels)
    #        dcf, _ = utils.DCF(pscores, labels, prior_t=prior)
    #        dcfs.append(dcf)
    #        ers.append(er)
    #        print(er)
    #    toprint.append((dcfs, ers, priors))
    #    latex(toprint)