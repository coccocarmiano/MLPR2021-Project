import numpy as np
import utils
from classifiers import logreg, logreg_scores

def latex(toprint):
    outfiletex = '../data/logreg_acc.tex'
    f = open(outfiletex, "w")
    print(r"\begin{center}", file=f)
    print(r"\begin{tabular}{|c|c|}", file=f)
    print(r"\hline", file=f)
    print(r"$\lambda$ & Error Rate \\", file=f)
    print(r"\hline", file=f)
    
    for tup in toprint:
        print(f"{tup[0]} & {(1-tup[1])*100:.3f}\\% \\\\", file=f)
        print(r"\hline", file=f)

    print(r"\hline", file=f)
    print(r"\end{tabular}", file=f)
    print(r"\end{center}", file=f)
    print(r"\caption{Logistic Regression}\label{tab:logreg}", file=f)

if __name__ == '__main__':
    trdataset = utils.load_train_data()
    tedataset = utils.load_test_data()
    toprint = []
    lambdas = [0, 10**-9, 10**-6, 10**-3, 0.1]
    for l in lambdas:
        print(f"Computing for lambda: {l}")
        w, b = logreg(trdataset, l)
        scores, predictions, acc = logreg_scores(tedataset, w, b)
        toprint.append((l, acc))
    
    latex(toprint)
