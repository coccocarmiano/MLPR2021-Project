import numpy as np
import utils
from classifiers import RBF_SVM

def latex(toprint):
    outfiletex = '../data/svm_rbf_acc.tex'
    f = open(outfiletex, "w")
    print(r"\begin{center}", file=f)
    print(r"\begin{longtable}{|c|c|c|c|}", file=f)
    print(r"\caption{Kernel SVM}\label{tab:svm_rbf_acctable}\\", file=f)
    print(r"\hline", file=f)
    print(r"Bias & $\gamma$ & $C$ & Error Rate \\", file=f)
    print(r"\hline", file=f)
    
    toprint.sort(key=lambda x: x[3], reverse=True)
    for tup in toprint[:10]:
        print(f"{tup[0]} & {tup[1]} & {tup[2]} & {(1-tup[3])*100:.3f}\\% \\\\", file=f)
        print(r"\hline", file=f)

    print(r"\hline", file=f)
    print(r"\end{longtable}", file=f)
    print(r"\end{center}", file=f)

if __name__ == '__main__':
    trdataset = utils.load_train_data()
    tedataset = utils.load_test_data()
    toprint = []
    biases = [0, 1.]
    boundaries = [.1, 1.]
    gammas = [5, 10]
    for bias in biases:
        for gamma in gammas:
            for boundary in boundaries:
                scores, predictions, acc = RBF_SVM(
                    trdataset, tedataset, gamma=gamma, reg_bias=bias, boundary=boundary)
                toprint.append((bias, gamma, boundary, acc))
    
    latex(toprint)
