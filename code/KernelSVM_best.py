import numpy as np
import utils
from classifiers import DualSVM_Train, DualSVM_Score

def kerf(regbias, gamma):
    def f(x1, x2):
        diff = x1-x2
        value = (diff * diff).sum() * gamma * -1
        value = np.exp(value)
        return value + regbias
    return f

if __name__ == '__main__':
    train, test = utils.load_train_data(), utils.load_test_data()
    trs, trl = train[:-1, :], train[-1, :]
    tes, tel = test[:-1, :], test[-1, :]
    trsn, testn = utils.gaussianize(trs, other=tes)
    train, test = np.vstack((trsn, trl)), np.vstack((testn, tel))
    print(train.shape, test.shape)
    opt_thresh, gamma, regbias, bias = 41.035, 0.15, .005, 15

    alphas = DualSVM_Train(train)

    f = kerf(regbias, gamma)
    scores = DualSVM_Score(train, alphas, test, function=f, bias = bias)
    preds = scores > opt_thresh

    nc = (preds == tel).sum()
    nt = len(preds)
    acc = nc/nt*100
    print(f"Accuracy: {acc:.3f}")