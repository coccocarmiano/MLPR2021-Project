import numpy as np
import utils
from classifiers import SVM_lin, SVM_lin_scores

def latex(toprint):
    outfiletex = '../data/svm_linpcan_acc.tex'
    f = open(outfiletex, "w")
    print(r"\begin{center}", file=f)
    print(r"\begin{tabular}{|c|c|c|c|}", file=f)
    print(r"\hline", file=f)
    print(r"Bias & $C$ & PCA & Error Rate \\", file=f)
    print(r"\hline", file=f)
    
    toprint.sort(key=lambda x: x[3], reverse=True)
    for tup in toprint[:10]:
        print(f"{tup[0]} & {tup[1]} & {tup[2]} & {(1-tup[3])*100:.3f}\\% \\\\", file=f)
        print(r"\hline", file=f)

    print(r"\end{tabular}", file=f)
    print(r"\end{center}", file=f)
    print(r"\caption{Linear SVM With PCA and Z-Normalization}\label{tab:svm_linpcan_acctable}", file=f)

if __name__ == '__main__':
    trdataset = utils.load_train_data()
    tedataset = utils.load_test_data()

    trsamp, trlab = trdataset[:-1, :], trdataset[-1, :]
    tesamp, telab = tedataset[:-1, :], tedataset[-1, :]
    trsamp, tesamp = utils.normalize(trsamp, other=tesamp)

    w, v = utils.PCA(trsamp, feat_label=False)

    toprint = []
    biases = [0, 1, 5., 10.]
    boundaries = [.1, 1]
    nPCA = [10, 8, 6, 4]
    for bias in biases:
        for boundary in boundaries:
            for n in nPCA:
                print(f"Calculating model with parameters (PCA: {n}): K={bias}, C={boundary}")
                vt = v[:, :n]
                ptrsamp, ptesamp = vt.T @ trsamp, vt.T @ tesamp
                trdata = np.vstack([ptrsamp, trlab])
                tedata = np.vstack([ptesamp, telab])
                w, b = SVM_lin(trdata, bias, boundary)
                scores, predictions, acc = SVM_lin_scores(tedata, w, b)
                toprint.append((bias, boundary, n, acc))
    
    latex(toprint)