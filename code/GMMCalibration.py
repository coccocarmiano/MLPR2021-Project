import utils
import numpy as np

if __name__ == '__main__':
    gmm_norm_scores = np.load('../data/GMMNorm-PCA11-3ComponentsScore.npy')
    gmm_scores = np.load('../data/GMM-PCA10-3ComponentsScore.npy')
    labels = utils.load_train_data()[-1, :]

    gmm_norm_dcf, _ = utils.DCF(gmm_norm_scores > 0, labels)
    gmm_dcf, _ = utils.DCF(gmm_scores > 0, labels)

    gmm_norm_mindcf, _ = utils.minDCF(gmm_norm_scores, labels)
    gmm_mindcf, _ = utils.minDCF(gmm_scores, labels)

    gmm_norm_scores_calibrated = utils.calibrate_scores(gmm_norm_scores, labels, .5)
    gmm_scores_calibrated = utils.calibrate_scores(gmm_scores, labels, .5)

    gmm_norm_calibrated_dcf, _ = utils.DCF(gmm_norm_scores_calibrated > 0, labels)
    gmm_calibrated_dcf, _ = utils.DCF(gmm_scores_calibrated > 0, labels)

    with open('GMMCalibration.txt', 'w') as f:
        print(f"minDCF {gmm_norm_mindcf} DCF Before {gmm_norm_dcf} DCF After {gmm_norm_calibrated_dcf}", file=f)
        print(f"minDCF {gmm_mindcf} DCF Before {gmm_dcf} DCF After {gmm_calibrated_dcf}", file=f)
        f.close()



