import numpy as np
from numpy import pi, log, exp, diag
from classifiers import gaussian_classifier
import utils

def latex(toprint):
    outfiletex = '../data/mvg_naiveacctable.tex'
    f = open(outfiletex, "w")
    print(r"\begin{center}", file=f)
    print(r"\begin{longtable}{|c|c|c|}", file=f)
    print(r"\caption{Naive Bayes}\label{tab:mvg_naiveacctable}\\", file=f)
    print(r"\hline", file=f)
    print(r"$\pi_T$ & PCA & Error Rate\\", file=f)
    print(r"\hline", file=f)
    
    for tup in toprint:
        print(f"{tup[0]} & {tup[1]} & {(1-tup[2])*100:.3f}\\% \\\\", file=f)
        print(r"\hline", file=f)

    print(r"\hline", file=f)
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

            ptrdata = vt.T.dot(trsamp)
            gsamples, bsamples = ptrdata[:, trlab > 0], ptrdata[:, trlab < 1]
            gmean, bmean = utils.fc_mean(gsamples), utils.fc_mean(bsamples)
            gcov, bcov = utils.fc_cov(gsamples), utils.fc_cov(bsamples)
            gcov, bcov = diag(diag(gcov)), diag(diag(bcov))

            ptedata = vt.T.dot(tesamp)
            scores, predictions = gaussian_classifier(ptedata, [gmean, bmean], [gcov, bcov])
            nt = len(predictions)
            nc = (predictions == telab).sum()
            acc = nc/nt
            
            if acc > 0.5:
                toprint.append((prior, n, acc))


    latex(toprint)
