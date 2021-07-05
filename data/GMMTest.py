import numpy as np
import utils
import scipy
from classifiers import gaussian_ll
import os



    

if __name__ == '__main__':
    train = utils.load_train_data()
    test = utils.load_test_data()
    trs, trl = train[:-1, :], train[-1, :]
    tes, tel = test[:-1, :], test[-1, :]

    train_good, train_bad = trs[:, trl > 0], trs[:, trl < 1]
    good_params = GMM_Train(train_good, 4)
    bad_params = GMM_Train(train_bad, 4)

    good_scores = GMM_Score(tes, good_params[0], good_params[1], good_params[2])
    bad_scores = GMM_Score(tes, bad_params[0], bad_params[1], bad_params[2])

    scores = good_scores - bad_scores
    predictions = scores > 0
    acc = (predictions == tel).sum() / len(tel) * 100
    print(f"Acc: {acc:.2f}")
    