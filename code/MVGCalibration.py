import utils
import numpy as np

if __name__ == '__main__':
    fullcov_scores = np.load('../data/MVGScoresPCA9.npy')
    naive_scores = np.load('../data/MVGNaiveScoresPCA7.npy')
    labels = utils.load_train_data()[-1, :]

    fullcov_dcf, _ = utils.DCF(fullcov_scores > 0, labels)
    naive_dcf, _ = utils.DCF(naive_scores > 0, labels)

    fullcov_mindcf, _ = utils.minDCF(fullcov_scores, labels, thresholds=fullcov_scores)
    naive_mindcf, _ = utils.minDCF(naive_scores, labels, thresholds=naive_scores)

    fullcov_calibrated_scores = utils.calibrate_scores(fullcov_scores, labels, .5)
    naive_calibrated_scores = utils.calibrate_scores(naive_scores, labels, .5)

    fullcov_dcf_after, _ = utils.DCF(fullcov_calibrated_scores > 0, labels)
    naive_dcf_after, _ = utils.DCF(naive_calibrated_scores > 0, labels)

    with open('MVGCalibration.txt', 'w') as f:
        print(f"minDCF : {fullcov_mindcf:.3f} -- DCF Before {fullcov_dcf:.3f} -- DCF After {fullcov_dcf_after:.3f} -- ", file=f)
        print(f"minDCF : {naive_mindcf:.3f} -- DCF Before {naive_dcf:.3f} -- DCF After {naive_dcf_after:.3f} -- ", file=f)