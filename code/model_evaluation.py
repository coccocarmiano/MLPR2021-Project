import utils
import matplotlib.pyplot as plt
from classifiers import SVM_lin, SVM_lin_scores, logreg, logreg_scores
from LogRegQuad import expand_feature_space
import calibration
import numpy as np


def plot_eval(scores, tel, calibrated, t, tag, filename):
    tag = tag + ' Bayes Error Plot'
    plt.figure(figsize=(7, 4))
    plt.title(tag)
    (dcfs, x), (mindcfs, x) = utils.BEP(scores, tel)
    plt.plot(x, dcfs, label="DCF")
    plt.plot(x, mindcfs, linestyle='--', label="DCF min")
    (dcfs, _), _ = utils.BEP(calibrated, tel)
    plt.plot(x, dcfs, linestyle='--', label="DCF (Log.Reg. Cal.)")
    a = np.linspace(.01, .99, 100)
    tups = [ utils.DCF(scores > t, tel, prior_t=p) for p in a ]
    tups = np.array([ a for a, _ in tups])
    plt.plot(x, tups, linestyle='--', label="DCF Val. Thresh.")
    plt.ylim((0.2, 1.1))
    plt.xlim((-2, 2))
    plt.ylabel('DCF', size=16)
    plt.xlabel(r'$t = -\log\frac{\pi_T}{1-\pi_T}$', size=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'../img/{filename}BEP.jpg')

p = 0.5

train = utils.load_train_data()
test = utils.load_test_data()

# LogReg
# Better results on validation set without calibration
l = 0.01
dim = 9

trdataset, tedataset = utils.normalize(train, other=test)
trdataset, tedataset = utils.reduce_dataset(trdataset, other=tedataset, n=dim)
trlabels = trdataset[-1]
telabels = tedataset[-1]

w, b = logreg(trdataset, l)
trscores, _, _ = logreg_scores(trdataset, w, b)
tescores, _, _ = logreg_scores(tedataset, w, b)

tescores_cal = calibration.calibrate_scores(trscores, trlabels, tescores, p)

_, t = utils.minDCF(trscores, trlabels)
mindcf, _ = utils.minDCF(tescores, telabels)
actdcf, _ = utils.DCF(tescores > 0, telabels)
print(f"logreg | actDCF {actdcf:.3f}, minDCF {mindcf:.3f}")
plot_eval(tescores, telabels, tescores_cal, t, 'Linear Logistic Regression', 'logreg')
# LogRegQuad
# Better results on validation set with calibration


## Normalization
l = 0.001
dim = 10

trdataset, tedataset = utils.normalize(train, other=test)
trdataset, tedataset = utils.reduce_dataset(trdataset, other=tedataset, n=dim)
trdataset, tedataset = expand_feature_space(trdataset), expand_feature_space(tedataset)
trlabels = trdataset[-1]
telabels = tedataset[-1]

w, b = logreg(trdataset, l)
trscores, _, _ = logreg_scores(trdataset, w, b)
tescores, _, _ = logreg_scores(tedataset, w, b)

tescores_cal = calibration.calibrate_scores(trscores, trlabels, tescores, p)
actdcf, _ = utils.DCF(tescores_cal > 0, telabels)
mindcf, _ = utils.minDCF(tescores_cal, telabels)
_, t = utils.minDCF(trscores, trlabels)
print(f"logregquad | norm,calibrated | actDCF {actdcf:.3f}, minDCF {mindcf:.3f}")
plot_eval(tescores, telabels, tescores_cal, t, 'Quadratic Logistic Regression', 'logregquad')


# Linear SVM
# Better results on validation set without calibration

# Normalized & 10 & 0.1 & 9 & 0.350 &  {\bf 0.330}  \\
K = 10
C = 0.1
dim = 9

train, test = utils.load_train_data(), utils.load_test_data()
trdataset, tedataset = utils.normalize(train, other=test)
trdataset, tedataset = utils.reduce_dataset(trdataset, other=tedataset, n=dim)
trdataset, tedataset = expand_feature_space(trdataset), expand_feature_space(tedataset)
trlabels = trdataset[-1]
telabels = tedataset[-1]

w, b = SVM_lin(trdataset, K, C)
trscores, _, _ = SVM_lin_scores(trdataset, w, b)
tescores, _, _ = SVM_lin_scores(tedataset, w, b)

tescores_cal = calibration.calibrate_scores(trscores, trlabels, tescores, p)
actdcf, _ = utils.DCF(tescores_cal > 0, telabels)
mindcf, _ = utils.minDCF(tescores_cal, telabels)
_, t = utils.minDCF(trscores, trlabels)
print(f"linsvm | norm,calibrated | actDCF {actdcf:.3f}, minDCF {mindcf:.3f}")
plot_eval(tescores, telabels, tescores_cal, t, 'Linear SVM', 'linsvm')