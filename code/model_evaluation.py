import utils
from classifiers import SVM_lin, SVM_lin_scores, logreg, logreg_scores
from LogRegQuad import expand_feature_space
import calibration
import numpy as np

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
tescores, _, _ = logreg_scores(tedataset, w, b)

actdcf, _ = utils.DCF(tescores > 0, telabels)
mindcf, _ = utils.minDCF(tescores, telabels)
print(f"logreg | actDCF {actdcf:.3f}, minDCF {mindcf:.3f}")

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

tescores = calibration.calibrate_scores(trscores, trlabels, tescores, p)
actdcf, _ = utils.DCF(tescores > 0, telabels)
mindcf, _ = utils.minDCF(tescores, telabels)
print(f"logregquad | norm,calibrated | actDCF {actdcf:.3f}, minDCF {mindcf:.3f}")

## Whitening
_, v = utils.whiten(train)
trfeats, trlabels = train[:-1], train[-1]
trfeats = v.T @ trfeats
trdataset = np.vstack((trfeats, trlabels))

tefeats, telabels = test[:-1], test[-1]
tefeats = v.T @ tefeats
tedataset = np.vstack((tefeats, telabels))

trdataset, tedataset = utils.reduce_dataset(trdataset, other=tedataset, n=dim)
trdataset, tedataset = expand_feature_space(trdataset), expand_feature_space(tedataset)
trlabels = trdataset[-1]
telabels = tedataset[-1]

w, b = logreg(trdataset, l)
trscores, _, _ = logreg_scores(trdataset, w, b)
tescores, _, _ = logreg_scores(tedataset, w, b)

tescores = calibration.calibrate_scores(trscores, trlabels, tescores, p)
actdcf, _ = utils.DCF(tescores > 0, telabels)
mindcf, _ = utils.minDCF(tescores, telabels)
print(f"logregquad | whiten,calibrated | actDCF {actdcf:.3f}, minDCF {mindcf:.3f}")

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

tescores = calibration.calibrate_scores(trscores, trlabels, tescores, p)
actdcf, _ = utils.DCF(tescores > 0, telabels)
mindcf, _ = utils.minDCF(tescores, telabels)
print(f"linsvm | norm,calibrated | actDCF {actdcf:.3f}, minDCF {mindcf:.3f}")