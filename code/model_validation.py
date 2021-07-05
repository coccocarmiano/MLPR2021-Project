import utils
import LinearSVM, LogReg, LogRegQuad
from calibration import calibrate_scores
from sys import stdout
import matplotlib.pyplot as plt
import numpy as np
trainedBase = '../trained'
classifiers = {'linsvm': LinearSVM, 'logreg': LogReg, 'logregquad': LogRegQuad}
outfile = stdout

def evaluate_model(data, classifier, tag='standard'):
    result = data[0]
    nPCA = sorted(result.keys(), reverse=True)
    p = 0.5

    if classifier == 'linsvm':
        biases, boundaries = data[1], data[2]
    else:
        lambdas = data[1]
    best = []
    for n in nPCA:
            if classifier == 'linsvm':
                for i, bias in enumerate(biases):
                    for j, bound in enumerate(boundaries):
                        scores = result[n][i, j, 0, :]
                        labels = result[n][i, j, 1, :]
                        # Not calibrated
                        mindcf, _ = utils.minDCF(scores, labels, p)
                        actdcf, _ = utils.DCF(scores > 0, labels)
                        best.append((mindcf, n, bias, bound, scores, labels, False))
                        print(f"{mindcf} |.| [{classifier}|{tag}] ActDCF: {actdcf:.4f} MinDCF: {mindcf:.4f}  -  PCA: {n}  -  Bias: {bias:.2f}  -  C: {bound:.2f}", file=outfile)
                        # Calibrated
                        scores = calibrate_scores(scores, labels, scores, p)
                        mindcf, _ = utils.minDCF(scores, labels, p)
                        actdcf, _ = utils.DCF(scores > 0, labels)
                        best.append((mindcf, n, bias, bound, scores, labels, True))
                        print(f"{mindcf} |.| [{classifier}|{tag}|calibrated] ActDCF: {actdcf:.4f} MinDCF: {mindcf:.4f}  -  PCA: {n}  -  Bias: {bias:.2f}  -  C: {bound:.2f}", file=outfile)
            else:
                for i, lam in enumerate(lambdas):
                    scores = result[n][i, 0, :]
                    labels = result[n][i, 1, :]
                    mindcf, _ = utils.minDCF(scores, labels, p)
                    actdcf, _ = utils.DCF(scores > 0, labels)
                    best.append((mindcf, n, lam, scores, labels, False))
                    print(f"{mindcf} |.| [{classifier}|{tag}] ActDCF: {actdcf:.4f} MinDCF: {mindcf:.4f}  -  PCA: {n}  -  Lambda: {lam}", file=outfile)
                    scores = calibrate_scores(scores, labels, scores, p)
                    mindcf, _ = utils.minDCF(scores, labels, p)
                    actdcf, _ = utils.DCF(scores > 0, labels)
                    best.append((mindcf, n, lam, scores, labels, False))
                    print(f"{mindcf} |.| [{classifier}|{tag}|calibrated] ActDCF: {actdcf:.4f} MinDCF: {mindcf:.4f}  -  PCA: {n}  -  Lambda: {lam}", file=outfile)


if __name__ == '__main__':

    for c in classifiers:
        evaluate_model(classifiers[c].result, c)
        evaluate_model(classifiers[c].result_norm, c, 'norm')
        evaluate_model(classifiers[c].result_whiten, c, 'whiten')