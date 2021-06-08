import numpy as np
import utils
from classifiers import SVM_lin, SVM_lin_scores

def latex(toprint):
    outfiletex = '../data/svm_linear_acc.tex'
    f = open(outfiletex, "w")
    print(r"\begin{center}", file=f)
    print(r"\begin{tabular}{|c|c|c|}", file=f)
    print(r"\hline", file=f)
    print(r"Bias & $C$ & Error Rate \\", file=f)
    print(r"\hline", file=f)
    
    for tup in toprint:
        print(f"{tup[0]} & {tup[1]} & {(1-tup[2])*100:.3f}\\% \\\\", file=f)
        print(r"\hline", file=f)

    print(r"\hline", file=f)
    print(r"\end{tabular}", file=f)
    print(r"\end{center}", file=f)
    print(r"\caption{Linear SVM}\label{tab:svm_linear_acctable}", file=f)

if __name__ == '__main__':
    trdataset = utils.load_train_data()
    tedataset = utils.load_test_data()
    toprint = []
    biases = [0, 1, 5., 10.]
    boundaries = [.1, 1]
    for bias in biases:
            for boundary in boundaries:
                print(f"Calculating model with parameters: K={bias}, C={boundary}")
                w, b = SVM_lin(trdataset, bias, boundary)
                scores, predictions, acc = SVM_lin_scores(tedataset, w, b)
                toprint.append((bias, boundary, acc))
    
    latex(toprint)