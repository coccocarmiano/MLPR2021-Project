import numpy as np
import classifiers
import utils

if __name__ == '__main__':
    narray = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    ## Raw Poly
    """ train = utils.load_train_data()
    w, v = utils.PCA(train)
    _, folds = utils.kfold(train, n=3)


    def f(x1, x2):
        # Dummy polynomial with c = 1 d = 2 b = .5
        val = (x1*x2).sum()
        val = (val + 1 ) ** 2
        return val + .5
    
    file = open('../data/PCAPolySVM.txt', 'w')
    for n in narray:
        scores, labels = [], []
        vt = v[:, :n]
        for fold in folds:
            trfold, tefold = fold
            trfold = np.vstack((vt.T @ trfold[:-1, :], trfold[-1, :]))
            tefold = np.vstack((vt.T @ tefold[:-1, :], tefold[-1, :]))
            labels.append(tefold[-1, :])
            alphas = classifiers.DualSVM_Train(trfold, f)
            trfold, alphas = utils.support_vectors(trfold, alphas)
            fold_scores = classifiers.DualSVM_Score(trfold, f, alphas, tefold)
            scores.append(fold_scores)
        
        scores, labels = np.concatenate(scores), np.concatenate(labels)
        dcf, _ = utils.DCF(scores > 0, labels)
        mindcf, _ = utils.minDCF(scores, labels)
        print(f"{dcf} || DCF: {dcf} -- minDCF {mindcf} -- PCA: {n}", file=file)
        np.save(f'../data/PCAPolySVMScoresPCA{n}.npy', scores)
    file.close()


    ## Normalized Poly
    train = utils.load_train_data()
    train = utils.normalize(train)
    w, v = utils.PCA(train)
    _, folds = utils.kfold(train, n=3)


    def f(x1, x2):
        # Dummy polynomial with c = 1 d = 2 b = .5
        val = (x1*x2).sum()
        val = (val + 1 ) ** 2
        return val + .5
    
    file = open('../data/PCAPolySVMNormalized.txt', 'w')
    for n in narray:
        scores, labels = [], []
        vt = v[:, :n]
        for fold in folds:
            trfold, tefold = fold
            trfold = np.vstack((vt.T @ trfold[:-1, :], trfold[-1, :]))
            tefold = np.vstack((vt.T @ tefold[:-1, :], tefold[-1, :]))
            labels.append(tefold[-1, :])
            alphas = classifiers.DualSVM_Train(trfold, f)
            trfold, alphas = utils.support_vectors(trfold, alphas)
            fold_scores = classifiers.DualSVM_Score(trfold, f, alphas, tefold)
            scores.append(fold_scores)
        
        scores, labels = np.concatenate(scores), np.concatenate(labels)
        dcf, _ = utils.DCF(scores > 0, labels)
        mindcf, _ = utils.minDCF(scores, labels)
        print(f"{dcf} || DCF: {dcf} -- minDCF {mindcf} -- PCA: {n}", file=file)
        np.save(f'../data/PCAPolySVMNormalizedScoresPCA{n}.npy', scores)
    file.close()


    ## Whitened Poly
    train = utils.load_train_data()
    train = utils.normalize(train)
    w, v = utils.whiten(train)
    _, folds = utils.kfold(train, n=3)


    def f(x1, x2):
        # Dummy polynomial with c = 1 d = 2 b = .5
        val = (x1*x2).sum()
        val = (val + 1 ) ** 2
        return val + .5
    
    file = open('../data/PCAPolySVMWhitened.txt', 'w')
    for n in narray:
        scores, labels = [], []
        vt = v[:, :n]
        for fold in folds:
            trfold, tefold = fold
            trfold = np.vstack((vt.T @ trfold[:-1, :], trfold[-1, :]))
            tefold = np.vstack((vt.T @ tefold[:-1, :], tefold[-1, :]))
            labels.append(tefold[-1, :])
            alphas = classifiers.DualSVM_Train(trfold, f)
            trfold, alphas = utils.support_vectors(trfold, alphas)
            fold_scores = classifiers.DualSVM_Score(trfold, f, alphas, tefold)
            scores.append(fold_scores)
        
        scores, labels = np.concatenate(scores), np.concatenate(labels)
        dcf, _ = utils.DCF(scores > 0, labels)
        mindcf, _ = utils.minDCF(scores, labels)
        print(f"{dcf} || DCF: {dcf} -- minDCF {mindcf} -- PCA: {n}", file=file)
        np.save(f'../data/PCAPolySVMWhitenedScoresPCA{n}.npy', scores)
    file.close()


    ## Raw RBF Kernel
    train = utils.load_train_data()
    w, v = utils.PCA(train)
    _, folds = utils.kfold(train, n=3)


    def f(x1, x2):
        # Dummy kernel with gamm .05 and regbias .05
        diff = (x1-x2)
        sq = (diff * diff).sum()
        sq = np.exp(sq * - .05) + 0.05
        return sq
    
    file = open('../data/PCARBFVM.txt', 'w')
    for n in narray:
        scores, labels = [], []
        vt = v[:, :n]
        for fold in folds:
            trfold, tefold = fold
            trfold = np.vstack((vt.T @ trfold[:-1, :], trfold[-1, :]))
            tefold = np.vstack((vt.T @ tefold[:-1, :], tefold[-1, :]))
            labels.append(tefold[-1, :])
            alphas = classifiers.DualSVM_Train(trfold, f)
            trfold, alphas = utils.support_vectors(trfold, alphas)
            fold_scores = classifiers.DualSVM_Score(trfold, f, alphas, tefold)
            scores.append(fold_scores)
        
        scores, labels = np.concatenate(scores), np.concatenate(labels)
        dcf, _ = utils.DCF(scores > 0, labels)
        mindcf, _ = utils.minDCF(scores, labels)
        print(f"{dcf} || DCF: {dcf} -- minDCF {mindcf} -- PCA: {n}", file=file)
        np.save(f'../data/PCARBFSVMScoresPCA{n}.npy', scores)
    file.close()


    ## Normalized RBF
    train = utils.load_train_data()
    train = utils.normalize(train)
    w, v = utils.PCA(train)
    _, folds = utils.kfold(train, n=3)


    def f(x1, x2):
        # Dummy kernel with gamm .05 and regbias .05
        diff = (x1-x2)
        sq = (diff * diff).sum()
        sq = np.exp(sq * - .05) + 0.05
        return sq
    
    file = open('../data/PCARBFNormalized.txt', 'w')
    for n in narray:
        scores, labels = [], []
        vt = v[:, :n]
        for fold in folds:
            trfold, tefold = fold
            trfold = np.vstack((vt.T @ trfold[:-1, :], trfold[-1, :]))
            tefold = np.vstack((vt.T @ tefold[:-1, :], tefold[-1, :]))
            labels.append(tefold[-1, :])
            alphas = classifiers.DualSVM_Train(trfold, f)
            trfold, alphas = utils.support_vectors(trfold, alphas)
            fold_scores = classifiers.DualSVM_Score(trfold, f, alphas, tefold)
            scores.append(fold_scores)
        
        scores, labels = np.concatenate(scores), np.concatenate(labels)
        dcf, _ = utils.DCF(scores > 0, labels)
        mindcf, _ = utils.minDCF(scores, labels)
        print(f"{dcf} || DCF: {dcf} -- minDCF {mindcf} -- PCA: {n}", file=file)
        np.save(f'../data/PCARBFSVMNormalizedScoresPCA{n}.npy', scores)
    file.close()


    ## Whitened RBF
    train = utils.load_train_data()
    train = utils.normalize(train)
    w, v = utils.whiten(train)
    _, folds = utils.kfold(train, n=3)


    def f(x1, x2):
        # Dummy kernel with gamm .05 and regbias .05
        diff = (x1-x2)
        sq = (diff * diff).sum()
        sq = np.exp(sq * - .05) + 0.05
        return sq
    
    file = open('../data/PCARBFWhitened.txt', 'w')
    for n in narray:
        scores, labels = [], []
        vt = v[:, :n]
        for fold in folds:
            trfold, tefold = fold
            trfold = np.vstack((vt.T @ trfold[:-1, :], trfold[-1, :]))
            tefold = np.vstack((vt.T @ tefold[:-1, :], tefold[-1, :]))
            labels.append(tefold[-1, :])
            alphas = classifiers.DualSVM_Train(trfold, f)
            trfold, alphas = utils.support_vectors(trfold, alphas)
            fold_scores = classifiers.DualSVM_Score(trfold, f, alphas, tefold)
            scores.append(fold_scores)
        
        scores, labels = np.concatenate(scores), np.concatenate(labels)
        dcf, _ = utils.DCF(scores > 0, labels)
        mindcf, _ = utils.minDCF(scores, labels)
        print(f"{dcf} || DCF: {dcf} -- minDCF {mindcf} -- PCA: {n}", file=file)
        np.save(f'../data/PCARBFSVMWhitenedScoresPCA{n}.npy', scores)
    file.close() """


    ## MVG
    train = utils.load_train_data()
    w, v = utils.PCA(train)
    _, folds = utils.kfold(train, n=3)

    file = open('../data/MVGPCA.txt', 'w')
    for n in narray:
        scores, labels = [], []
        vt = v[:, :n]
        for fold in folds:
            trfold, tefold = fold
            trsamples, trlabels = vt.T @ trfold[:-1, :], trfold[-1, :]
            tesamples, telabels = vt.T @ tefold[:-1, :], tefold[-1, :]

            good, bad = trsamples[:, trlabels > 0], trsamples[:, trlabels < 1]
            gmean, bmean = utils.fc_mean(good), utils.fc_mean(bad)
            gcov, bcov = utils.fc_cov(good), utils.fc_cov(bad)
            fold_scores = classifiers.gaussian_classifier(tesamples, [bmean, gmean], [bcov, gcov])
            scores.append(fold_scores)
            labels.append(telabels)
        
        scores, labels = np.concatenate(scores), np.concatenate(labels)
        dcf, _ = utils.DCF(scores > 0, labels)
        mindcf, _ = utils.minDCF(scores, labels)
        print(f"{dcf} || DCF: {dcf} -- minDCF {mindcf} -- PCA: {n}", file=file)
        np.save(f'../data/MVGScoresPCA{n}.npy', scores)
    file.close()


    ## MVG + Norm
    train = utils.load_train_data()
    train = utils.normalize(train)
    w, v = utils.PCA(train)
    _, folds = utils.kfold(train, n=3)

    file = open('../data/MVGPCANormalized.txt', 'w')
    for n in narray:
        scores, labels = [], []
        vt = v[:, :n]
        for fold in folds:
            trfold, tefold = fold
            trsamples, trlabels = vt.T @ trfold[:-1, :], trfold[-1, :]
            tesamples, telabels = vt.T @ tefold[:-1, :], tefold[-1, :]

            good, bad = trsamples[:, trlabels > 0], trsamples[:, trlabels < 1]
            gmean, bmean = utils.fc_mean(good), utils.fc_mean(bad)
            gcov, bcov = utils.fc_cov(good), utils.fc_cov(bad)
            fold_scores = classifiers.gaussian_classifier(tesamples, [bmean, gmean], [bcov, gcov])
            scores.append(fold_scores)
            labels.append(telabels)
        
        scores, labels = np.concatenate(scores), np.concatenate(labels)
        dcf, _ = utils.DCF(scores > 0, labels)
        mindcf, _ = utils.minDCF(scores, labels)
        print(f"{dcf} || DCF: {dcf} -- minDCF {mindcf} -- PCA: {n}", file=file)
        np.save(f'../data/MVGNormalizedScoresPCA{n}.npy', scores)
    file.close()

    ## MVG Whitened
    train = utils.load_train_data()
    train = utils.normalize(train)
    w, v = utils.whiten(train)
    _, folds = utils.kfold(train, n=3)

    file = open('../data/MVGPCAWhitened.txt', 'w')
    for n in narray:
        scores, labels = [], []
        vt = v[:, :n]
        for fold in folds:
            trfold, tefold = fold
            trsamples, trlabels = vt.T @ trfold[:-1, :], trfold[-1, :]
            tesamples, telabels = vt.T @ tefold[:-1, :], tefold[-1, :]

            good, bad = trsamples[:, trlabels > 0], trsamples[:, trlabels < 1]
            gmean, bmean = utils.fc_mean(good), utils.fc_mean(bad)
            gcov, bcov = utils.fc_cov(good), utils.fc_cov(bad)
            fold_scores = classifiers.gaussian_classifier(tesamples, [bmean, gmean], [bcov, gcov])
            scores.append(fold_scores)
            labels.append(telabels)
        
        scores, labels = np.concatenate(scores), np.concatenate(labels)
        dcf, _ = utils.DCF(scores > 0, labels)
        mindcf, _ = utils.minDCF(scores, labels)
        print(f"{dcf} || DCF: {dcf} -- minDCF {mindcf} -- PCA: {n}", file=file)
        np.save(f'../data/MVGWhitenedScoresPCA{n}.npy', scores)
    file.close()


    ## MVG Tied Cov
    train = utils.load_train_data()
    w, v = utils.PCA(train)
    _, folds = utils.kfold(train, n=3)

    file = open('../data/MVGTiedPCA.txt', 'w')
    for n in narray:
        scores, labels = [], []
        vt = v[:, :n]
        for fold in folds:
            trfold, tefold = fold
            trsamples, trlabels = vt.T @ trfold[:-1, :], trfold[-1, :]
            tesamples, telabels = vt.T @ tefold[:-1, :], tefold[-1, :]

            good, bad = trsamples[:, trlabels > 0], trsamples[:, trlabels < 1]
            gmean, bmean = utils.fc_mean(good), utils.fc_mean(bad)
            cov = utils.fc_cov(trsamples[:-1, :])
            gcov, bcov = cov, cov
            fold_scores = classifiers.gaussian_classifier(tesamples, [bmean, gmean], [bcov, gcov])
            scores.append(fold_scores)
            labels.append(telabels)
        
        scores, labels = np.concatenate(scores), np.concatenate(labels)
        dcf, _ = utils.DCF(scores > 0, labels)
        mindcf, _ = utils.minDCF(scores, labels)
        print(f"{dcf} || DCF: {dcf} -- minDCF {mindcf} -- PCA: {n}", file=file)
        np.save(f'../data/MVGTiedScoresPCA{n}.npy', scores)
    file.close()


    ## MVG Tied Cov + Norm
    train = utils.load_train_data()
    train = utils.normalize(train)
    w, v = utils.PCA(train)
    _, folds = utils.kfold(train, n=3)

    file = open('../data/MVGTiedNormalizedPCA.txt', 'w')
    for n in narray:
        scores, labels = [], []
        vt = v[:, :n]
        for fold in folds:
            trfold, tefold = fold
            trsamples, trlabels = vt.T @ trfold[:-1, :], trfold[-1, :]
            tesamples, telabels = vt.T @ tefold[:-1, :], tefold[-1, :]

            good, bad = trsamples[:, trlabels > 0], trsamples[:, trlabels < 1]
            gmean, bmean = utils.fc_mean(good), utils.fc_mean(bad)
            cov = utils.fc_cov(trsamples[:-1, :])
            gcov, bcov = cov, cov
            fold_scores = classifiers.gaussian_classifier(tesamples, [bmean, gmean], [bcov, gcov])
            scores.append(fold_scores)
            labels.append(telabels)
        
        scores, labels = np.concatenate(scores), np.concatenate(labels)
        dcf, _ = utils.DCF(scores > 0, labels)
        mindcf, _ = utils.minDCF(scores, labels)
        print(f"{dcf} || DCF: {dcf} -- minDCF {mindcf} -- PCA: {n}", file=file)
        np.save(f'../data/MVGTiedNormalizedScoresPCA{n}.npy', scores)
    file.close()

    ## MVG Tied Cov + Whitening
    train = utils.load_train_data()
    train = utils.normalize(train)
    w, v = utils.whiten(train)
    _, folds = utils.kfold(train, n=3)

    file = open('../data/MVGTiedWhitenedPCA.txt', 'w')
    for n in narray:
        scores, labels = [], []
        vt = v[:, :n]
        for fold in folds:
            trfold, tefold = fold
            trsamples, trlabels = vt.T @ trfold[:-1, :], trfold[-1, :]
            tesamples, telabels = vt.T @ tefold[:-1, :], tefold[-1, :]

            good, bad = trsamples[:, trlabels > 0], trsamples[:, trlabels < 1]
            gmean, bmean = utils.fc_mean(good), utils.fc_mean(bad)
            cov = utils.fc_cov(trsamples[:-1, :])
            gcov, bcov = cov, cov
            fold_scores = classifiers.gaussian_classifier(tesamples, [bmean, gmean], [bcov, gcov])
            scores.append(fold_scores)
            labels.append(telabels)
        
        scores, labels = np.concatenate(scores), np.concatenate(labels)
        dcf, _ = utils.DCF(scores > 0, labels)
        mindcf, _ = utils.minDCF(scores, labels)
        print(f"{dcf} || DCF: {dcf} -- minDCF {mindcf} -- PCA: {n}", file=file)
        np.save(f'../data/MVGTiedWhitenedScoresPCA{n}.npy', scores)
    file.close()



    ## MVG Naive Bayes
    train = utils.load_train_data()
    w, v = utils.PCA(train)
    _, folds = utils.kfold(train, n=3)

    file = open('../data/MVGNaivePCA.txt', 'w')
    for n in narray:
        scores, labels = [], []
        vt = v[:, :n]
        for fold in folds:
            trfold, tefold = fold
            trsamples, trlabels = vt.T @ trfold[:-1, :], trfold[-1, :]
            tesamples, telabels = vt.T @ tefold[:-1, :], tefold[-1, :]

            good, bad = trsamples[:, trlabels > 0], trsamples[:, trlabels < 1]
            gmean, bmean = utils.fc_mean(good), utils.fc_mean(bad)
            gcov, bcov = utils.fc_cov(good), utils.fc_cov(bad)
            gcov, bcov = np.diag(np.diag(gcov)), np.diag(np.diag(bcov))
            fold_scores = classifiers.gaussian_classifier(tesamples, [bmean, gmean], [bcov, gcov])
            scores.append(fold_scores)
            labels.append(telabels)
        
        scores, labels = np.concatenate(scores), np.concatenate(labels)
        dcf, _ = utils.DCF(scores > 0, labels)
        mindcf, _ = utils.minDCF(scores, labels)
        print(f"{dcf} || DCF: {dcf} -- minDCF {mindcf} -- PCA: {n}", file=file)
        np.save(f'../data/MVGNaiveScoresPCA{n}.npy', scores)
    file.close()


    ## MVG Naive Bayes + Norm
    train = utils.load_train_data()
    train = utils.normalize(train)
    w, v = utils.PCA(train)
    _, folds = utils.kfold(train, n=3)

    file = open('../data/MVGNaiveNormalizedPCA.txt', 'w')
    for n in narray:
        scores, labels = [], []
        vt = v[:, :n]
        for fold in folds:
            trfold, tefold = fold
            trsamples, trlabels = vt.T @ trfold[:-1, :], trfold[-1, :]
            tesamples, telabels = vt.T @ tefold[:-1, :], tefold[-1, :]

            good, bad = trsamples[:, trlabels > 0], trsamples[:, trlabels < 1]
            gmean, bmean = utils.fc_mean(good), utils.fc_mean(bad)
            gcov, bcov = utils.fc_cov(good), utils.fc_cov(bad)
            gcov, bcov = np.diag(np.diag(gcov)), np.diag(np.diag(bcov))
            fold_scores = classifiers.gaussian_classifier(tesamples, [bmean, gmean], [bcov, gcov])
            scores.append(fold_scores)
            labels.append(telabels)
        
        scores, labels = np.concatenate(scores), np.concatenate(labels)
        dcf, _ = utils.DCF(scores > 0, labels)
        mindcf, _ = utils.minDCF(scores, labels)
        print(f"{dcf} || DCF: {dcf} -- minDCF {mindcf} -- PCA: {n}", file=file)
        np.save(f'../data/MVGNaiveNormalizedScoresPCA{n}.npy', scores)
    file.close()

    ## MVG Naive Bayes + Whitening
    train = utils.load_train_data()
    train = utils.normalize(train)
    w, v = utils.whiten(train)
    _, folds = utils.kfold(train, n=3)

    file = open('../data/MVGNaiveWhitenedPCA.txt', 'w')
    for n in narray:
        scores, labels = [], []
        vt = v[:, :n]
        for fold in folds:
            trfold, tefold = fold
            trsamples, trlabels = vt.T @ trfold[:-1, :], trfold[-1, :]
            tesamples, telabels = vt.T @ tefold[:-1, :], tefold[-1, :]

            good, bad = trsamples[:, trlabels > 0], trsamples[:, trlabels < 1]
            gmean, bmean = utils.fc_mean(good), utils.fc_mean(bad)
            gcov, bcov = utils.fc_cov(good), utils.fc_cov(bad)
            gcov, bcov = np.diag(np.diag(gcov)), np.diag(np.diag(bcov))
            fold_scores = classifiers.gaussian_classifier(tesamples, [bmean, gmean], [bcov, gcov])
            scores.append(fold_scores)
            labels.append(telabels)
        
        scores, labels = np.concatenate(scores), np.concatenate(labels)
        dcf, _ = utils.DCF(scores > 0, labels)
        mindcf, _ = utils.minDCF(scores, labels)
        print(f"{dcf} || DCF: {dcf} -- minDCF {mindcf} -- PCA: {n}", file=file)
        np.save(f'../data/MVGNaiveWhitenedScoresPCA{n}.npy', scores)
    file.close()
    