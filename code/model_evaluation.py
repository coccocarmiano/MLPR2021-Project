import utils
import LinearSVM, LogReg, LogRegQuad
from calibration import calibrate_scores
from sys import stdout
import matplotlib.pyplot as plt
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
                        mindcf, optimal_threshold = utils.minDCF(scores, labels, p)
                        best.append((mindcf, n, bias, bound, scores, labels))
                        print(f"{mindcf} |.| [{classifier}|{tag}] MinDCF: {mindcf:.4f}  -  Opt. Thr.: {optimal_threshold:.4f}  -  Bias: {bias:.2f}  -  C: {bound:.2f}", file=outfile)
            else:
                for i, lam in enumerate(lambdas):
                    scores = result[n][i, 0, :]
                    labels = result[n][i, 1, :]
                    mindcf, optimal_threshold = utils.minDCF(scores, labels, p)
                    best.append((mindcf, n, lam, scores, labels))
                    print(f"{mindcf} |.| [{classifier}|{tag}] MinDCF: {mindcf:.4f}  -  Opt. Thr.: {optimal_threshold:.4f}  -  Lambda: {lam}", file=outfile)

    best = sorted(best, key=lambda x: x[0])[0]
    if classifier == 'linsvm':
        scores = best[4]
        labels = best[5]
    else:
        scores = best[3]
        labels = best[4]
    (actdcf_points, xaxis), (mindcf_points, xaxis) = utils.BEP(scores, labels)
    
    plt.figure()
    plt.title(f"{classifier} [{tag}]")
    plt.plot(xaxis, mindcf_points, 'r--')
    plt.plot(xaxis, actdcf_points, 'b')
    plt.xlim([min(xaxis), max(xaxis)])
    plt.ylim([0, 1.5])

    scores = calibrate_scores(scores, labels, p)
    (actdcf_points, xaxis), (mindcf_points, xaxis) = utils.BEP(scores, labels)

    if tag == 'standard':
        plt.figure()
        plt.title(f"{classifier} [{tag}] (calibrated)")
        plt.plot(xaxis, mindcf_points, 'r--')
        plt.plot(xaxis, actdcf_points, 'b')
        plt.xlim([min(xaxis), max(xaxis)])
        plt.ylim([0, 1.5])
    
    plt.show()


if __name__ == '__main__':

    for c in classifiers:
        evaluate_model(classifiers[c].result, c)
        evaluate_model(classifiers[c].result_norm, c, 'norm')
        evaluate_model(classifiers[c].result_gau, c, 'gau')
        evaluate_model(classifiers[c].result_whiten, c, 'whiten')