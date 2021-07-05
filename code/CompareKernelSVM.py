import utils
import numpy as np

if __name__ == '__main__':
    poly_norm_scores = np.load('../data/PolySVMNormalized-PCA10-C1-POW2-Bounds0.1Scores.npy')
    poly_raw_scores = np.load('../data/PolySVM-PCA9-C1-POW3-Bounds0.1Scores.npy')

    rbf_norm_scores = np.load('../data/KernelSVMNormalized-PCA10-RegBias0.1-Gamma0.05-Bound1.5Scores.npy')
    rbf_raw_scores = np.load('../data/KernelSVM-PCA11-RegBias0.1-Gamma0.05-Bound1.5Scores.npy')

    labels = utils.load_train_data()[-1, :]

    cal = utils.calibrate_scores(poly_norm_scores, labels, .5)
    dcf_before, _ = utils.DCF(poly_norm_scores > 0, labels)
    mindcf, _ = utils.minDCF(poly_norm_scores, labels)
    dcf_after, _ = utils.DCF(cal > 0, labels)
    print(f"Normalized Poly minDCF {mindcf} DCF Before {dcf_before} DCF After {dcf_after}")

    cal = utils.calibrate_scores(poly_raw_scores, labels, .5)
    dcf_before, _ = utils.DCF(poly_raw_scores > 0, labels)
    mindcf, _ = utils.minDCF(poly_raw_scores, labels)
    dcf_after, _ = utils.DCF(cal > 0, labels)
    print(f"Raw Poly minDCF {mindcf} DCF Before {dcf_before} DCF After {dcf_after}")

    cal = utils.calibrate_scores(rbf_norm_scores, labels, .5)
    dcf_before, _ = utils.DCF(rbf_norm_scores > 0, labels)
    mindcf, _ = utils.minDCF(rbf_norm_scores, labels)
    dcf_after, _ = utils.DCF(cal > 0, labels)
    print(f"Normalized RBF minDCF {mindcf} DCF Before {dcf_before} DCF After {dcf_after}")

    cal = utils.calibrate_scores(rbf_raw_scores, labels, .5)
    dcf_before, _ = utils.DCF(rbf_raw_scores > 0, labels)
    mindcf, _ = utils.minDCF(rbf_raw_scores, labels)
    dcf_after, _ = utils.DCF(cal > 0, labels)
    print(f"Raw RBF minDCF {mindcf} DCF Before {dcf_before} DCF After {dcf_after}")

