import numpy as np
from numpy import pi, log, exp
from classifiers import gaussian_classifier
import utils

def latex(toprint):
    outfiletex = '../data/mvg_acctable.tex'
    f = open(outfiletex, "w")
    print(r"\begin{center}", file=f)
    print(r"\begin{longtable}{|c|c|c|}", file=f)
    print(r"\caption{MVG}\label{tab:mvg_acctable}\\", file=f)
    print(r"\hline", file=f)
    print(r"$\pi_T$ & PCA & Error Rate\\", file=f)
    print(r"\hline", file=f)
    
    for tup in toprint:
        print(f"{tup[0]} & {tup[1]} & {(1-tup[2])*100:.3f}\\% \\\\", file=f)
        print(r"\hline", file=f)

    print(r"\hline", file=f)
    print(r"\end{longtable}", file=f)
    print(r"\end{center}", file=f)

if __name__ == '__main__':
    toprint = [] #ignore this var
    trdataset = utils.load_train_data()
    trsamp, trlab = trdataset[:-1, :], trdataset[-1, :]

    tedataset = utils.load_test_data()
    tesamp, telab = tedataset[:-1, :], tedataset[-1, :]

    nfeats = trdataset.shape[0]-1
    it = np.arange(nfeats)
    it = it[-1:0:-1]

    w, v = utils.PCA(tedataset)
    priors = (np.arange(9)+1)/10

    
    for prior in priors:
        for n in it:
            vt = v[:, :n]
            pdata = vt.T.dot(tesamp)
            gmean, bmean = utils.fc_mean(pdata[:, telab > 0]), utils.fc_mean(pdata[:, telab < 1])
            gcov, bcov = utils.fc_cov(pdata[:, telab > 0]), utils.fc_cov(pdata[:, telab < 1])
            scores, labels = gaussian_classifier(pdata, [gmean, bmean], [gcov, bcov], prior_t=prior)
            nc = (labels == telab).sum()
            nt = len(telab)
            acc = nc/nt
            if acc > 0.6:
                toprint.append((prior, n, acc))

    latex(toprint)
