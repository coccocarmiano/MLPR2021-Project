import numpy as np
import utils
from classifiers import logreg, logreg_scores

def latex(toprint):
    outfiletex = '../data/logreg_pca_acc.tex'
    f = open(outfiletex, "w")
    print(r"\begin{center}", file=f)
    print(r"\begin{tabular}{|c|c|c|}", file=f)
    print(r"\hline", file=f)
    print(r"$\lambda$ & PCA & Error Rate \\", file=f)
    print(r"\hline", file=f)
    
    toprint.sort(key=lambda x: x[2], reverse=True)
    for tup in toprint[:10]:
        print(f"{tup[0]} & {tup[1]} & {(1-tup[2])*100:.3f}\\% \\\\", file=f)
        print(r"\hline", file=f)

    print(r"\end{tabular}", file=f)
    print(r"\end{center}", file=f)
    print(r"\caption{Logistic Regression With PCA}\label{tab:logreg_pca_acctable}", file=f)

if __name__ == '__main__':
    trdataset = utils.load_train_data()
    tedataset = utils.load_test_data()

    trsamp, trlab = trdataset[:-1, :], trdataset[-1, :]
    tesamp, telab = tedataset[:-1, :], tedataset[-1, :]

    w, v = utils.PCA(trsamp, feat_label=False)

    toprint = []
    lambdas = [0, 10**-9, 10**-6, 10**-3, 0.1]
    nPCA = [10, 8, 6, 4]
    for l in lambdas:
        for n in nPCA:
            print(f"Calculating model with parameters (PCA: {n}): l = {l}")
            vt = v[:, :n]
            ptrsamp, ptesamp = vt.T @ trsamp, vt.T @ tesamp
            trdata = np.vstack([ptrsamp, trlab])
            tedata = np.vstack([ptesamp, telab])
            w, b = logreg(trdata, l)
            scores, predictions, acc = logreg_scores(tedata, w, b)
            toprint.append((l, n, acc))
    
    latex(toprint)