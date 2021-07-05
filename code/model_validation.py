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

    #best = sorted(best, key=lambda x: x[0])[0]
    #plt.figure()
    #if classifier == 'linsvm':
    #    scores = best[4]
    #    labels = best[5]
    #    plt.title(f"{classifier} [{tag}] (PCA: {best[1]}, Bias: {best[2]}, Bound: {best[3]})")
    #else:
    #    scores = best[3]
    #    labels = best[4]
    #    plt.title(f"{classifier} [{tag}] (PCA: {best[1]}, λ: {best[2]})")
    #
    #(actdcf_points, xaxis), (mindcf_points, xaxis) = utils.BEP(scores, labels)
#
    #plt.plot(xaxis, mindcf_points, 'r--', label='minDCF')
    #plt.plot(xaxis, actdcf_points, 'b', label="actDCF")
    #plt.xlim([min(xaxis), max(xaxis)])
    #plt.ylim([0, 1.5])
#
    #if tag == 'standard':
    #    scores = calibrate_scores(scores, labels, scores, p)
    #    (calibrated_actdcf_points, xaxis), (_, _) = utils.BEP(scores, labels)
    #    plt.plot(xaxis, calibrated_actdcf_points, color='tab:green', label="actDCF (calibrated)")
    #    plt.ylim([0, 1.5])
    #
    ##plt.show()
    #plt.legend()
    #plt.savefig(f"../img/BEP_{classifier}_{tag}.jpg")


if __name__ == '__main__':

    for c in classifiers:
        evaluate_model(classifiers[c].result, c)
        evaluate_model(classifiers[c].result_norm, c, 'norm')
        evaluate_model(classifiers[c].result_gau, c, 'gau')
        evaluate_model(classifiers[c].result_whiten, c, 'whiten')