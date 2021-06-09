import numpy as np
from numpy import diag
from classifiers import gaussian_classifier
from matplotlib import pyplot as plt
import utils

#
#def tablehead(file, title, label, format="|c|c||c|c|"):
#    print(f"\\caption{title}\\label{label}", file=file)
#    print(r"\begin{center}", file=file)
#    print(f"\\begin{{tabular}}{format}", file=file)
#    print(r"\hline", file=file)
#    print(r"\ & PCA & Error Rate & $DCF$\\", file=file)
#    print(r"\hline", file=file)
#
#
#def tabletail(file):
#    print(r"\end{tabular}", file=file)
#    print(r"\end{center}", file=file)
#    file.close()
#
#
#def latex(toprint):
#    outfiletex = '../data/mvg_acctable.tex'
#    f = open(outfiletex, "w")
#    tablehead(f, "title", "label")
#
#    toprint.sort(key=lambda x: min(x[0]))
#    toprint = toprint[:3]
#    for tup in toprint:
#        for i in range(len(tup[0])):
#            print(
#                f"$\\pi_T = {tup[3][i]:.2f}$ & {tup[1]} & {tup[2][i]*100:.2f} & {tup[0][i]:.3f} \\\\", file=f)
#        print(r"\hline", file=f)
#
#    tabletail(f)

printsettings = {
    'method' : 'Full-Covariance',
    'figname' : 'mvgfullcov.jpg',
    'figlabel' : 'tab:mvgfullcov',
}


def fullcov(folds, priors, npca, eigv):
    results = []
    for n in npca:
        scores, labels = np.empty(0), np.empty(0)
        vt = eigv[:, :n]
        for fold in folds:
            trs, trl = fold[0][:-1, :], fold[0][-1, :]
            tes, tel = fold[1][:-1, :], fold[1][-1, :]
            ptes = vt.T @ tes

            gmean = utils.fc_mean(trs[:, trl > 0])
            bmean = utils.fc_mean(trs[:, trl < 1])
            gcov = utils.fc_cov(trs[:, trl > 0])
            bcov = utils.fc_cov(trs[:, trl < 1])

            pgmean = vt.T @ gmean
            pbmean = vt.T @ bmean

            pgcov = vt.T @ gcov @ vt
            pbcov = vt.T @ bcov @ vt

            fscores, _ = gaussian_classifier(ptes, [pbmean, pgmean], [pbcov, pgcov])
            scores = np.hstack((scores, fscores))
            labels = np.hstack((labels, tel))
        
        mindcf, pi, points = utils.minDCF(scores, labels, priors)
        results.append({'npca' : n, 'mindcf' : mindcf, 'prior' : pi, 'points' : points})

    plot_mindcf(results)
    table_mindcf(results)


def naive(folds, priors, npca, eigv):
    results = []
    for n in npca:
        scores, labels = np.empty(0), np.empty(0)
        vt = eigv[:, :n]
        for fold in folds:
            trs, trl = fold[0][:-1, :], fold[0][-1, :]
            tes, tel = fold[1][:-1, :], fold[1][-1, :]
            ptes = vt.T @ tes

            gmean = utils.fc_mean(trs[:, trl > 0])
            bmean = utils.fc_mean(trs[:, trl < 1])
            gcov = utils.fc_cov(trs[:, trl > 0])
            bcov = utils.fc_cov(trs[:, trl < 1])

            pgmean = diag(diag(vt.T @ gmean)) # Diff
            pbmean = diag(diag(vt.T @ bmean)) # Diff

            pgcov = vt.T @ gcov @ vt
            pbcov = vt.T @ bcov @ vt

            fscores, _ = gaussian_classifier(ptes, [pbmean, pgmean], [pbcov, pgcov])
            scores = np.hstack((scores, fscores))
            labels = np.hstack((labels, tel))
        
        mindcf, pi, points = utils.minDCF(scores, labels, priors)
        results.append({'npca' : n, 'mindcf' : mindcf, 'prior' : pi, 'points' : points})

    plot_mindcf(results)
    table_mindcf(results)


def tied(folds, priors, npca, eigv):
    results = []
    for n in npca:
        scores, labels = np.empty(0), np.empty(0)
        vt = eigv[:, :n]
        for fold in folds:
            trs, trl = fold[0][:-1, :], fold[0][-1, :]
            tes, tel = fold[1][:-1, :], fold[1][-1, :]
            ptes = vt.T @ tes

            gmean = utils.fc_mean(trs[:, trl > 0])
            bmean = utils.fc_mean(trs[:, trl < 1])
            cov = utils.fc_cov(trs) # Diff

            pgmean = vt.T @ gmean
            pbmean = vt.T @ bmean

            pcov = vt.T @ cov @ vt

            fscores, _ = gaussian_classifier(ptes, [pbmean, pgmean], [pcov, pcov])
            scores = np.hstack((scores, fscores))
            labels = np.hstack((labels, tel))
        
        mindcf, pi, points = utils.minDCF(scores, labels, priors)
        results.append({'npca' : n, 'mindcf' : mindcf, 'prior' : pi, 'points' : points})

    plot_mindcf(results)
    table_mindcf(results)



def plot_mindcf(results):
    results.sort(key=lambda res: res['mindcf'])
    bestr = results[0]
    points = bestr['points']
    npca = bestr['npca']


    x = [i[0] for i in points]
    y = [i[1] for i in points]
    plt.plot(x, y)
    plt.title(f"{printsettings['method']} MVG DCF vs $\pi_T$ (PCA = {npca})")
    plt.xlabel(r"$-log(\frac{\pi_T}{1-\pi_T})$")
    plt.ylabel(r"$DCF$")

    plt.savefig(f'../img/figtitle.jpg')

def table_mindcf(results):
    results.sort(key=lambda res: res['mindcf'])
    top3 = results[:3]
    print(f"{printsettings['method']}")
    for res in top3:
        print(f"|\tPCA: {res['npca']}\t|\tPrior: {res['prior']:.3f}\t|\tminDCF: {res['mindcf']:3f}\t|")


if __name__ == '__main__':
    toprint = []  # ignore this var
    to_plot = []
    trdataset = utils.load_train_data()
    nfolds = 5

    _, v = utils.PCA(trdataset)
    _, folds = utils.kfold(trdataset, n=nfolds)

    priors = np.linspace(.01, .99, 1000)
    npca = np.arange(8)+3

    printsettings['method'] = 'Full Covariance'
    printsettings['figname'] = 'mvgfullcov.jpg'
    fullcov(folds, priors, npca, v)

    printsettings['method'] = 'Naive Bayes'
    printsettings['figname'] = 'mvgnaive.jpg'
    fullcov(folds, priors, npca, v)

    printsettings['method'] = 'Tied Covariance'
    printsettings['figname'] = 'mvgtied.jpg'
    fullcov(folds, priors, npca, v)

    feats, labels = trdataset[:-1, :], trdataset[-1, :]
    feats = utils.normalize(feats)
    trdataset = np.vstack((feats, labels))

    _, v = utils.PCA(trdataset)
    _, folds = utils.kfold(trdataset, n=nfolds)
    
    printsettings['method'] = 'Normalized Full Covariance'
    printsettings['figname'] = 'mvgfullcovnorm.jpg'
    fullcov(folds, priors, npca, v)

    printsettings['method'] = 'Normalized Naive Bayes'
    printsettings['figname'] = 'mvgnaivenorm.jpg'
    fullcov(folds, priors, npca, v)

    printsettings['method'] = 'Normalized Tied Covariance'
    printsettings['figname'] = 'mvgtiednorm.jpg'
    fullcov(folds, priors, npca, v)