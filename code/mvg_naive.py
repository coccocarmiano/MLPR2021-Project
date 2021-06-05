import numpy as np
from numpy import pi, log, exp, diag
from classifiers import gaussian_classifier
import utils

def latex_report(toprint):
    outfiletex = '../data/mvg_naiveacctable.tex'
    f = open(outfiletex, "w")
    print(r"\begin{center}", file=f)
    print(r"\begin{longtable}{|c|c|c|}", file=f)
    print(r"\caption{MVG Classifier Error Rates (Naive Bayes)}\label{tab:mvg_naiveacctable}\\", file=f)
    print(r"\hline", file=f)
    print(r"Prior & PCA & \% Errors \\", file=f)
    print(r"\hline", file=f)
    
    for line in toprint:
        print(line, file=f)

    print(r"\end{longtable}", file=f)
    print(r"\end{center}", file=f)

    f.close()

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
            gcov, bcov = diag(diag(gcov)), diag(diag(bcov))
            scores, labels = gaussian_classifier(pdata, [gmean, bmean], [gcov, bcov], prior_t=prior)
            nc = (labels == telab).sum()
            nt = len(telab)

###########Ignore this part
            toprint.append(f"{prior} & {n} & {(1-(nc/nt))*100:.3f}\\% \\\\")
            toprint.append(r"\hline")
        toprint.append(r"\hline")

    latex_report(toprint)
