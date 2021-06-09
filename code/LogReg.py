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


def plot_lambda_minDCF(dataset):
    lambdas = np.logspace(-5, 5, 11)
    dcfs = np.empty(len(lambdas))
    priors = [.1, .5, .9]
    brg = ['b', 'r', 'g']
    patches = [ptc.Patch(color=c, label=f"pi = {p}")for p, c in zip(priors, brg)]
    _, folds = utils.kfold(dataset, n=5)

    plt.figure()
    plt.xlabel('lambda')
    plt.xscale('log')
    plt.ylabel('minDCF')
    for i, p in enumerate(priors):
        for j, l in enumerate(lambdas):
            print(f"Computing for p = {p}, l = {l}")
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
        plt.plot(lambdas, dcfs, brg[i])
    plt.figlegend(handles=patches, handlelength=1, loc='upper right')
    plt.savefig('../img/logreg_lambda_minDCF.jpg', format='jpg')


if __name__ == '__main__':
    plot_lambda_minDCF(utils.load_train_data())
    def tmp():
        trdataset = utils.load_train_data()
        tedataset = utils.load_test_data()
        _, folds = utils.kfold(trdataset, n=5)
        toprint = []
        lambdas = [0, 10**-9, 10**-6, 10**-3, 0.1]
        priors = [.1, .50, .9]
        tot_scores = []
        tot_labels = []
        for fold in folds:
            trdata = fold[0]
            tedata = fold[1]
            telabel = fold[1][-1, :]
            for l in lambdas:
                print(f"Computing for lambda: {l}")
                w, b = logreg(trdata, l)
                scores, _, _ = logreg_scores(tedata, w, b)
                tot_scores.append(scores)
                tot_labels.append(telabel)
        scores = np.concatenate(tot_scores)
        labels = np.concatenate(tot_labels)
        
        dcfs, ers = [], []
        for prior in priors:
            t = - np.log( prior / (1-prior))
            pscores = scores > t
            er = (pscores != labels).sum() / len(labels)
            dcf, _ = utils.DCF(pscores, labels, prior_t=prior)
            dcfs.append(dcf)
            ers.append(er)
            print(er)
        toprint.append((dcfs, ers, priors))
        latex(toprint)
