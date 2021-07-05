import utils
import classifiers
import numpy as np
from matplotlib import pyplot as plt

# Consult Report and /data/ folder for hyperparameters

if __name__ == '__main__':


    # Tied Covariance Model
    train, test = utils.load_train_data(), utils.load_test_data()
    train, test = utils.normalize(train, other=test)
    w, v = utils.PCA(train)
    vt = v[:, :5]
    trs, trl = vt.T @ train[:-1, :], train[-1, :]
    tes, tel = vt.T @ test[:-1, :], test[-1, :]

    gmean, bmean = utils.fc_mean(trs[:, trl > 0]), utils.fc_mean(trs[:, trl < 1])
    cov = utils.fc_cov(trs)
    scores = classifiers.gaussian_classifier(tes, [bmean, gmean], [cov, cov])
    dcf, _ = utils.DCF(scores > 0, tel)
    mindcf, _ = utils.minDCF(scores, tel)
    temp = np.load('../data/MVG-TIED-NORMALIZED-PCA5.npy')
    _, t = utils.minDCF(temp, trl)
    dcf2, _ = utils.DCF(scores > t, tel)
    alpha, beta = utils.calibrate_scores_params(scores, tel, .5)
    calibrated = alpha * scores + beta # -log(...)
    dcf3, _ = utils.DCF(calibrated > 0, tel)
    er = dcf * 50
    print(f"Tied Covariance || PCA 7 ~ minDCF {mindcf} DCF {dcf} DCFValidation {dcf2} DCFLogreg {dcf3} ER {er}")

    plt.figure(figsize=(14, 8))
    plt.title('Tied Covariance Bayes Error Plot')
    (dcfs, x), (mindcfs, x) = utils.BEP(scores, tel)
    plt.plot(x, dcfs, label="DCF")
    plt.plot(x, mindcfs, linestyle='--', label="DCF min")
    (dcfs, _), _ = utils.BEP(calibrated, tel)
    plt.plot(x, dcfs, linestyle='--', label="DCF (Log.Reg. Cal.)")
    a = np.linspace(.01, .99, 100)
    tups = [ utils.DCF(scores > t, tel, prior_t=p) for p in a ]
    tups = [ a for a, _ in tups]
    plt.plot(x, tups, linestyle='--', label="DCF Val. Thresh.")
    plt.ylim((0.2, 1.1))
    plt.xlim((-2, 2))
    plt.ylabel('DCF', size=16)
    plt.xlabel(r'$t = -\log\frac{\pi_T}{1-\pi_T}$', size=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig('../img/MVGTiedBEP.jpg')
    exit()

    # GMM 

    train, test = utils.load_train_data(), utils.load_test_data()
    trs, trl = train[:-1, :], train[-1, :]
    tes, tel = test[:-1, :], test[-1, :]

    gw, gm, gc = classifiers.GMM_Train(trs[:, trl > 0], 2)
    bw, bm, bc = classifiers.GMM_Train(trs[:, trl < 1], 2)

    scores = classifiers.GMM_Score(tes, gw, gm, gc) - classifiers.GMM_Score(tes, bw, bm, bc)
    dcf, _ = utils.DCF(scores > 0, tel)
    mindcf, _ = utils.minDCF(scores, tel)
    temp = np.load('../data/GMM-RAW-2COMPONENTS-PCA11.npy')
    _, t = utils.minDCF(temp, trl)
    dcf2, _ = utils.DCF(scores > t, tel)
    alpha, beta = utils.calibrate_scores_params(scores, tel, .5)
    calibrated = alpha * scores + beta # -log(...)
    dcf3, _ = utils.DCF(calibrated > 0, tel)
    er = dcf * 50
    print(f"GMM || Norm. PCA NO 3 Components ~ minDCF {mindcf} DCF {dcf} DCFValidation {dcf2} DCFLogreg {dcf3} ER {er}")
    exit()


    plt.figure(figsize=(14, 8))
    plt.subplot(2, 1, 1).set_title("GMM, Normalized Features")
    (dcfs, x), (mindcfs, x) = utils.BEP(scores, tel)
    plt.plot(x, dcfs, label="DCF")
    plt.plot(x, mindcfs, linestyle='--', label="DCF min")
    (dcfs, _), _ = utils.BEP(calibrated, tel)
    plt.plot(x, dcfs, linestyle='--', label="DCF (Log.Reg. Cal.)")
    a = np.linspace(.01, .99, 100)
    tups = [ utils.DCF(scores > t, tel, prior_t=p) for p in a ]
    tups = [ a for a, _ in tups]
    plt.plot(x, tups, linestyle='--', label="DCF Val. Thresh.")
    plt.ylim((0.2, 1.1))
    plt.xlim((-2, 2))
    plt.ylabel('DCF', size=16)
    plt.legend()


    # GMM Raw

    train, test = utils.load_train_data(), utils.load_test_data()
    w, v = utils.PCA(train)
    vt = v[:, :10]
    trs, trl = vt.T @ train[:-1, :], train[-1, :]
    tes, tel = vt.T @ test[:-1, :], test[-1, :]

    gw, gm, gc = classifiers.GMM_Train(trs[:, trl > 0], 3)
    bw, bm, bc = classifiers.GMM_Train(trs[:, trl < 1], 3)

    scores = classifiers.GMM_Score(tes, gw, gm, gc) - classifiers.GMM_Score(tes, bw, bm, bc)
    dcf, _ = utils.DCF(scores > 0, tel)
    mindcf, _ = utils.minDCF(scores, tel)
    temp = np.load('../data/GMM-PCA10-3ComponentsScore.npy')
    _, t = utils.minDCF(temp, trl)
    dcf2, _ = utils.DCF(scores > t, tel)
    alpha, beta = utils.calibrate_scores_params(scores, tel, .5)
    calibrated = alpha * scores + beta # -log(...)
    dcf3, _ = utils.DCF(calibrated > 0, tel)
    er = dcf * 50
    print(f"GMM || Raw PCA 10 3 Components ~ minDCF {mindcf} DCF {dcf} DCFValidation {dcf2} DCFLogreg {dcf3} ER {er}")


    plt.subplot(2, 1, 2).set_title("GMM, Raw Features")
    (dcfs, x), (mindcfs, x) = utils.BEP(scores, tel)
    plt.plot(x, dcfs, label="DCF")
    plt.plot(x, mindcfs, linestyle='--', label="DCF min")
    (dcfs, _), _ = utils.BEP(calibrated, tel)
    plt.plot(x, dcfs, linestyle='--', label="DCF (Log.Reg. Cal.)")
    a = np.linspace(.01, .99, 100)
    tups = [ utils.DCF(scores > t, tel, prior_t=p) for p in a ]
    tups = [ a for a, _ in tups]
    plt.plot(x, tups, linestyle='--', label="DCF Val. Thresh.")
    plt.ylim((0.2, 1.1))
    plt.xlim((-2, 2))
    plt.ylabel('DCF', size=16)
    plt.xlabel(r'$t = -\log\frac{\pi_T}{1-\pi_T}$', size=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig('../img/GMMBEP.jpg')




    # Normalized RBF

    def frbf(x1, x2):
        diff = (x1-x2)
        diff = diff * diff
        value = diff.sum() * -.05
        value = np.exp(value)
        return value + 0.1

    train, test = utils.load_train_data(), utils.load_test_data()
    train, test = utils.normalize(train, other=test)
    w, v = utils.PCA(train)
    vt = v[:, :10]
    train = np.vstack((vt.T @ train[:-1, :], train[-1, :]))
    test = np.vstack((vt.T@ test[:-1, :], test[-1, :]))
    tel = test[-1, :]
    trl = train[-1, :]

    alphas = classifiers.DualSVM_Train(train, frbf, bound=1.5)
    train, alphas = utils.support_vectors(train, alphas)
    scores = classifiers.DualSVM_Score(train, frbf, alphas, test)
    dcf, _ = utils.DCF(scores > 0, tel)
    mindcf, _ = utils.minDCF(scores, tel)
    temp = np.load('../data/KernelSVMNormalized-PCA10-RegBias0.1-Gamma0.05-Bound1.5Scores.npy')
    _, t = utils.minDCF(temp, trl)
    dcf2, _ = utils.DCF(scores > t, tel)
    alpha, beta = utils.calibrate_scores_params(scores, tel, .5)
    calibrated = alpha * scores + beta # -log(...)
    dcf3, _ = utils.DCF(calibrated > 0, tel)
    er = dcf * 50
    print(f"RBFSVM || Norm PCA 10 Gamma = 0.05 RegBias = 0.1 Bound 1.5 ~ minDCF {mindcf} DCF {dcf} DCFValidation {dcf2} DCFLogreg {dcf3} ER {er}")


    plt.figure(figsize=(7, 4))
    plt.title('RBF Kernel Normalized Features Bayes Error Plot')
    (dcfs, x), (mindcfs, x) = utils.BEP(scores, tel)
    plt.plot(x, dcfs, label="DCF")
    plt.plot(x, mindcfs, linestyle='--', label="DCF min")
    (dcfs, _), _ = utils.BEP(calibrated, tel)
    plt.plot(x, dcfs, linestyle='--', label="DCF (Log.Reg. Cal.)")
    a = np.linspace(.01, .99, 100)
    tups = [ utils.DCF(scores > t, tel, prior_t=p) for p in a ]
    tups = [ a for a, _ in tups]
    plt.plot(x, tups, linestyle='--', label="DCF Val. Thresh.")
    plt.ylim((0.2, 1.1))
    plt.xlim((-2, 2))
    plt.ylabel('DCF', size=16)
    plt.xlabel(r'$t = -\log\frac{\pi_T}{1-\pi_T}$', size=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig('../img/RBFNormalizedBEP.jpg')
