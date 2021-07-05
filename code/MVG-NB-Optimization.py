import utils, classifiers
import numpy as np

if __name__ == '__main__':
    ## Raw
    dataset = utils.load_train_data()
    npca = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    w, v = utils.PCA(dataset)
    _, folds = utils.kfold(dataset, n=10)

    f = open('../data/MVG-NAIVE-RAW.txt', 'w')
    for n in npca:
        vt = v[:, :n]
        scores, labels = [], []
        for fold in folds:
            train, test = fold
            trs, trl = vt.T @ train[:-1, :], train[-1, :]
            tes, tel = vt.T @ test[:-1, :], test[-1, :]

            gmean, bmean = utils.fc_mean(trs[:, trl > 0]), utils.fc_mean(trs[:, trl < 1])
            gcov, bcov = utils.fc_cov(trs[:, trl > 0]), utils.fc_cov(trs[:, trl < 1])
            gcov, bcov = np.diag(np.diag(gcov)), np.diag(np.diag(bcov))

            fold_scores = classifiers.gaussian_classifier(tes, [bmean, gmean], [bcov, gcov])
            scores.append(fold_scores)
            labels.append(tel)
        scores, labels = np.concatenate(scores), np.concatenate(labels)
        dcf, _ = utils.DCF(scores > 0, labels)
        mindcf, _ = utils.minDCF(scores, labels)
        np.save(f'../data/MVG-NAIVE-RAW-PCA{n}.npy', scores)
        print(f"{mindcf} || DCF {dcf} minDCF {mindcf} PCA {n}", file=f)
    f.close()

    ## Normalized

    ## Raw
    dataset = utils.load_train_data()
    dataset = utils.normalize(dataset)
    w, v = utils.PCA(dataset)
    _, folds = utils.kfold(dataset, n=10)

    f = open('../data/MVG-NAIVE-NORMALIZED.txt', 'w')
    for n in npca:
        vt = v[:, :n]
        scores, labels = [], []
        for fold in folds:
            train, test = fold
            trs, trl = vt.T @ train[:-1, :], train[-1, :]
            tes, tel = vt.T @ test[:-1, :], test[-1, :]

            gmean, bmean = utils.fc_mean(trs[:, trl > 0]), utils.fc_mean(trs[:, trl < 1])
            gcov, bcov = utils.fc_cov(trs[:, trl > 0]), utils.fc_cov(trs[:, trl < 1])
            gcov, bcov = np.diag(np.diag(gcov)), np.diag(np.diag(bcov))

            fold_scores = classifiers.gaussian_classifier(tes, [bmean, gmean], [bcov, gcov])
            scores.append(fold_scores)
            labels.append(tel)
        scores, labels = np.concatenate(scores), np.concatenate(labels)
        dcf, _ = utils.DCF(scores > 0, labels)
        mindcf, _ = utils.minDCF(scores, labels)
        np.save(f'../data/MVG-NAIVE-NORMALIZED-PCA{n}.npy', scores)
        print(f"{mindcf} || DCF {dcf} minDCF {mindcf} PCA {n}", file=f)
    f.close()

    ## Whitened

    dataset = utils.load_train_data()
    w, v = utils.whiten(dataset)
    dataset = utils.normalize(dataset)
    _, folds = utils.kfold(dataset, n=10)

    f = open('../data/MVG-NAIVE-WHITENED.txt', 'w')
    for n in npca:
        vt = v[:, :n]
        scores, labels = [], []
        for fold in folds:
            train, test = fold
            trs, trl = vt.T @ train[:-1, :], train[-1, :]
            tes, tel = vt.T @ test[:-1, :], test[-1, :]

            gmean, bmean = utils.fc_mean(trs[:, trl > 0]), utils.fc_mean(trs[:, trl < 1])
            gcov, bcov = utils.fc_cov(trs[:, trl > 0]), utils.fc_cov(trs[:, trl < 1])
            gcov, bcov = np.diag(np.diag(gcov)), np.diag(np.diag(bcov))

            fold_scores = classifiers.gaussian_classifier(tes, [bmean, gmean], [bcov, gcov])
            scores.append(fold_scores)
            labels.append(tel)
        scores, labels = np.concatenate(scores), np.concatenate(labels)
        dcf, _ = utils.DCF(scores > 0, labels)
        mindcf, _ = utils.minDCF(scores, labels)
        np.save(f'../data/MVG-NAIVE-WHITENED-PCA{n}.npy', scores)
        print(f"{mindcf} || DCF {dcf} minDCF {mindcf} PCA {n}", file=f)
    f.close()


            
