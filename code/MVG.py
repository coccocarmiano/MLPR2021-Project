import utils, classifiers
import numpy as np

if __name__ == '__main__':
    dataset = utils.load_train_data()
    npca = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    ## Normalized

    normalized_dataset = utils.normalize(dataset)
    w, v = utils.PCA(normalized_dataset)
    _, folds = utils.kfold(normalized_dataset, n=20)

    for n in npca:
        vt = v[:, :n]
        scores, labels = [], []
        for fold in folds:
            train, test = fold
            trs, trl = vt.T @ train[:-1, :], train[-1, :]
            tes, tel = vt.T @ test[:-1, :], test[-1, :]

            gmean, bmean = utils.fc_mean(trs[:, trl > 0]), utils.fc_mean(trs[:, trl < 1])
            # gcov, bcov = utils.fc_cov(trs[:, trl > 0]), utils.fc_cov(trs[:, trl < 1])
            cov = utils.fc_cov(trs)

            fold_scores = classifiers.gaussian_classifier(tes, [bmean, gmean], [cov, cov])
            
            scores.append(fold_scores)
            labels.append(tel)
        
        labels, scores = np.concatenate(labels), np.concatenate(scores)
        dcf, _ = utils.DCF(scores > 0, labels)
        print(f"{dcf} MVG NORM PCA {n}")

    # Whitening
    dataset = utils.load_train_data()
    normalized_dataset = utils.normalize(dataset)
    w, v = utils.whiten(normalized_dataset)
    _, folds = utils.kfold(normalized_dataset, n=20)

    for n in npca:
        vt = v[:, :n]
        scores, labels = [], []
        for fold in folds:
            train, test = fold
            trs, trl = vt.T @ train[:-1, :], train[-1, :]
            tes, tel = vt.T @ test[:-1, :], test[-1, :]

            gmean, bmean = utils.fc_mean(trs[:, trl > 0]), utils.fc_mean(trs[:, trl < 1])
            # gcov, bcov = utils.fc_cov(trs[:, trl > 0]), utils.fc_cov(trs[:, trl < 1])
            cov = utils.fc_cov(trs)

            fold_scores = classifiers.gaussian_classifier(tes, [bmean, gmean], [cov, cov])
            
            scores.append(fold_scores)
            labels.append(tel)
        
        labels, scores = np.concatenate(labels), np.concatenate(scores)
        dcf, _ = utils.DCF(scores > 0, labels)
        print(f"{dcf} MVG WHI PCA {n}")

    print("Debug")