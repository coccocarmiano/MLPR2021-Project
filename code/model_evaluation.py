import numpy as np
import LinearSVM, LogReg, LogRegQuad
import utils
from sys import stdout
from os.path import join, isfile

trainedBase = '../trained'
classifiers = {'linsvm': LinearSVM, 'logreg': LogReg, 'logregquad': LogRegQuad}
outfile = stdout


def evaluate_model(data, classifier, tag='standard'):
    result = data[0]
    nPCA = sorted(result.keys(), reverse=True)
    
    if classifier == 'linsvm':
        biases, boundaries = data[1], data[2]
    else:
        lambdas = data[1]

    for n in nPCA:
        for p in [0.5]:
            if classifier == 'linsvm':
                for i, bias in enumerate(biases):
                    for j, bound in enumerate(boundaries):
                        scores = result[n][i, j, 0, :]
                        labels = result[n][i, j, 1, :]
                        mindcf, optimal_threshold = utils.minDCF(scores, labels, p)
                        print(f"{mindcf} |.| [{classifier}|{tag}] MinDCF: {mindcf:.4f}  -  Opt. Thr.: {optimal_threshold:.4f}  -  Bias: {bias:.2f}  -  C: {bound:.2f}", file=outfile)
            else:
                for i, lam in enumerate(lambdas):
                    scores = result[n][i, 0, :]
                    labels = result[n][i, 1, :]
                    mindcf, optimal_threshold = utils.minDCF(scores, labels, p)
                    print(f"{mindcf} |.| [{classifier}|{tag}] MinDCF: {mindcf:.4f}  -  Opt. Thr.: {optimal_threshold:.4f}  -  Lambda: {lam}", file=outfile)

if __name__ == '__main__':

    for c in classifiers:
        evaluate_model(classifiers[c].result, c)
        evaluate_model(classifiers[c].result_norm, c, 'norm')
        evaluate_model(classifiers[c].result_gau, c, 'gau')
        evaluate_model(classifiers[c].result_whiten, c, 'whiten')