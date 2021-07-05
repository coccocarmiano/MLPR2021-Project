import calibration
import utils
import classifiers
import numpy as np

if __name__ == '__main__':
    ncomponents = [2, 3, 4, 5, 6]
    npca = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    ## Raw

    dataset = utils.load_train_data()
    w, v = utils.PCA(dataset)
    _, folds = utils.kfold(dataset, n=10)

    f = open('../data/GMMRAW.txt', 'w')

    for n in npca:
        for nc in ncomponents:
            vt = v[:, :n]
            scores, labels = [], []
            for fold in folds:
                train, test = fold
                trs, trl = vt.T @ train[:-1, :], train[-1, :]
                tes, tel = vt.T @ test[:-1, :], test[-1, :]

                trs_good = trs[:, trl > 0]
                trs_bad = trs[:, trl < 1]

                gw, gm, gc = classifiers.GMM_Train(trs_good, nc)
                bw, bm, bc = classifiers.GMM_Train(trs_bad, nc)

                fold_scores = classifiers.GMM_Score(tes, gw, gm, gc) - classifiers.GMM_Score(tes, bw, bm, bc)
                scores.append(fold_scores)
                labels.append(tel)
            
            scores, labels = np.concatenate(scores), np.concatenate(labels)
            np.save(f"../data/GMM-RAW-{nc}COMPONENTS-PCA{n}.npy", scores)
            dcf, _ = utils.DCF(scores > 0, labels)
            mindcf, _ = utils.minDCF(scores, labels)
            print(f"{mindcf} || DCF {dcf} minDCF {mindcf} NComponents {nc} PCA {n}", file=f)
    f.close()

    ## Norm

    dataset = utils.load_train_data()
    dataset = utils.normalize(dataset)
    w, v = utils.PCA(dataset)
    _, folds = utils.kfold(dataset, n=10)

    f = open('../data/GMMNorm.txt', 'w')

    for n in npca:
        for nc in ncomponents:
            vt = v[:, :n]
            scores, labels = [], []
            for fold in folds:
                train, test = fold
                trs, trl = vt.T @ train[:-1, :], train[-1, :]
                tes, tel = vt.T @ test[:-1, :], test[-1, :]

                trs_good = trs[:, trl > 0]
                trs_bad = trs[:, trl < 1]

                gw, gm, gc = classifiers.GMM_Train(trs_good, nc)
                bw, bm, bc = classifiers.GMM_Train(trs_bad, nc)

                fold_scores = classifiers.GMM_Score(tes, gw, gm, gc) - classifiers.GMM_Score(tes, bw, bm, bc)
                scores.append(fold_scores)
                labels.append(tel)
            
            scores, labels = np.concatenate(scores), np.concatenate(labels)
            np.save(f"../data/GMM-Norm-{nc}COMPONENTS-PCA{n}.npy", scores)
            dcf, _ = utils.DCF(scores > 0, labels)
            mindcf, _ = utils.minDCF(scores, labels)
            print(f"{mindcf} || DCF {dcf} minDCF {mindcf} NComponents {nc} PCA {n}", file=f)
    f.close()

    ## Whitened

    dataset = utils.load_train_data()
    dataset = utils.normalize(dataset)
    w, v = utils.whiten(dataset)
    _, folds = utils.kfold(dataset, n=10)

    f = open('../data/GMMWHITENED.txt', 'w')

    for n in npca:
        for nc in ncomponents:
            vt = v[:, :n]
            scores, labels = [], []
            for fold in folds:
                train, test = fold
                trs, trl = vt.T @ train[:-1, :], train[-1, :]
                tes, tel = vt.T @ test[:-1, :], test[-1, :]

                trs_good = trs[:, trl > 0]
                trs_bad = trs[:, trl < 1]

                gw, gm, gc = classifiers.GMM_Train(trs_good, nc)
                bw, bm, bc = classifiers.GMM_Train(trs_bad, nc)

                fold_scores = classifiers.GMM_Score(tes, gw, gm, gc) - classifiers.GMM_Score(tes, bw, bm, bc)
                scores.append(fold_scores)
                labels.append(tel)
            
            scores, labels = np.concatenate(scores), np.concatenate(labels)
            np.save(f"../data/GMM-WHITENED-{nc}COMPONENTS-PCA{n}.npy", scores)
            dcf, _ = utils.DCF(scores > 0, labels)
            mindcf, _ = utils.minDCF(scores, labels)
            print(f"{mindcf} || DCF {dcf} minDCF {mindcf} NComponents {nc} PCA {n}", file=f)
    f.close()



    


                

