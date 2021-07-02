import utils
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    scores = np.load('../data/MVGTiedScoresPCA7.npy')
    labels = utils.load_train_data()[-1, :]
    adcf, mdcf = utils.BEP(scores, labels)
    plt.plot(adcf[1], adcf[0])
    plt.ylim((0, 1.2))
    plt.show()