import numpy as np
from numpy import pi, log, exp, diag
from classifiers import gaussian_classifier
import utils

def latex(toprint):
    outfiletex = '../data/mvg_tiedacctable.tex'
    f = open(outfiletex, "w")
    print(r"\caption{Tied Covariance MVG}\label{tab:mvg_tiedcov}", file=f)
    print(r"\begin{center}", file=f)
    print(r"\begin{tabular}{|c|c||c|c|}", file=f)
    print(r"\hline", file=f)
    print(r"\ & PCA & Error Rate \% & $DCF$\\", file=f)
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

if __name__ == '__main__':
    toprint = [] #ignore this var
    trdataset = utils.load_train_data()
    trlab = trdataset[-1, :]
    nt = len(trlab)

    _, folds = utils.kfold(trdataset)

    priors = [.33, .50, .67]
    npca = [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

   
    for nred in npca:
        tot_scores = []
        tot_labels = []
        for fold in folds:
            trs, trl = fold[0][:-1, :], fold[0][-1, :]
            tes, tel = fold[1][:-1, :], fold[1][-1, :]
            gmean, bmean = utils.fc_mean(trs[:, trl > 0]), utils.fc_mean(trs[:, trl < 1])
            cov = utils.fc_cov(trs)
            _, v = utils.PCA(trs, feat_label=False)
            vt = v[:, :nred]
            pgmean, pbmean = vt.T @ gmean, vt.T @ bmean
            pcov = vt.T @ cov @ vt
            ptes = vt.T @ tes
            scores, _ = gaussian_classifier(ptes, [pbmean, pgmean], [pcov, pcov])
            tot_scores.append(scores)
            tot_labels.append(tel)
        scores = np.concatenate(tot_scores)
        labels = np.concatenate(tot_labels)
        
        dcfs, ers = [], []
        for prior in priors:
            t = np.log(1-prior) - np.log(prior)
            pscores = scores > t
            er = (pscores != labels).sum() / len(labels)
            dcf, _ = utils.DCF(pscores, labels, prior_t=prior)
            dcfs.append(dcf)
            ers.append(er)
            print(er)
        toprint.append((dcfs, nred, ers, priors))
        

    latex(toprint)
