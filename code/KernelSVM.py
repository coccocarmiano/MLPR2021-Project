import numpy as np
import utils
from classifiers import RBF_SVM

def latex(toprint):
    outfiletex = '../data/svm_rbf_acc.tex'
    f = open(outfiletex, "w")
    print(r"\caption{Kernel SVM}\label{tab:svm_rbf_acctable}", file=f)
    print(r"\begin{center}", file=f)
    print(r"\begin{tabular}{|c|c|c|c|c|c|}", file=f)
    print(r"\hline", file=f)
    print(r"$\pi_t$ & Bias & \gamma & C & \% Error & $DCF_{min}$ \\", file=f)
    print(r"\hline", file=f)
    
    toprint.sort(key=lambda x: x[5])
    for tup in toprint[:10]:
        print(f"{tup[0]} & {tup[1]} & {tup[2]} & {tup[3]} & {(1-tup[4])*100:.2f} & {tup[5]:.3f}\\\\", file=f)
        print(r"\hline", file=f)

    print(r"\end{tabular}", file=f)
    print(r"\end{center}", file=f)

if __name__ == '__main__':
    trdataset = utils.load_train_data()
    _, folds = utils.kfold(trdataset)
    toprint = []
    biases = [1e0, 1e1, 1e2, 1e3, 1e4]
    boundaries = [.1, .5, 1.]
    gammas = [.1, .5, 1, 5, 10]
    priors = [0.33, 0.5, 0.67]

    for fold in folds:
        trs = fold[0]
        tes = fold[1]
        for prior in priors:
            for bias in biases:
                for gamma in gammas:
                    for boundary in boundaries:
                        tel = tes[-1, :]
                        scores, predictions, acc = RBF_SVM(
                            trs, tes, gamma=gamma, reg_bias=bias, boundary=boundary)
                        _, dcf = utils.DCF(predictions, tel, prior_t=prior)
                        nt = len(predictions)
                        nc = (predictions == tel).sum()
                        acc = nc/nt
                        toprint.append((prior, bias, gamma, boundary, acc, dcf))
    
    latex(toprint)
