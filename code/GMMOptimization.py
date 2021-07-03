import calibration
import utils
import classifiers
import numpy as np

if __name__ == '__main__':
    ncomponents = [2, 3, 4, 5]
    npca = [11, 10, 9, 8]
    dataset = utils.load_train_data()
    dataset = utils.normalize(dataset)
    w, v = utils.PCA(dataset)
    _, folds = utils.kfold(dataset)

    file = open('../data/GMMNormOptimization.txt', 'w')

    for nc in ncomponents:
        for p in npca:
            """ if p == 11 and nc == 5:
                continue """
            labels, scores = [], []
            vt = v[:, :p]
            for fold in folds:
                train, test = fold

                trs, trl = train[:-1, :], train[-1, :]
                tes, tel = test[:-1, :], test[-1, :]

                train = vt.T @ trs
                train_good, train_bad = train[:, trl > 0], train[:, trl < 1]
                test = vt.T @ tes

                gweights, gmeans, gcovs = classifiers.GMM_Train(train_good, nc)
                bweights, bmeans, bcovs = classifiers.GMM_Train(train_bad, nc)

                llr = classifiers.GMM_Score(test, gweights, gmeans, gcovs)-classifiers.GMM_Score(test, bweights, bmeans, bcovs)

                scores.append(llr)
                labels.append(tel)
            
            scores, labels = np.concatenate(scores), np.concatenate(labels)
            dcf, _ = utils.DCF(scores > 0, labels)
            mindcf, _ = utils.minDCF(scores, labels, thresholds=scores)
            calibrated = calibration.calibrate_scores(scores, labels, .5)
            dcf_calibrated, _ = utils.DCF(calibrated > 0, labels)
            print(f"{dcf} || PCA {p} -- DCF {dcf} -- minDCF {mindcf} DCF_Cal {dcf_calibrated} -- Nc {nc}", file=file)
            print(f"{dcf} || PCA {p} -- DCF {dcf} -- minDCF {mindcf} DCF_Cal {dcf_calibrated} -- Nc {nc}")
            np.save(f'../data/GMMNorm-PCA{p}-{nc}ComponentsScore.npy', scores)

    print("")


                

