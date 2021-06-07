import numpy as np
import utils
from matplotlib import pyplot as plt

fout = '../img/scatter.jpg'
nfeats = 11

if __name__ == '__main__':
    data = utils.load_train_data()
    w, v = utils.PCA(data)
    w, v = w[:2], v[:, :2]

    data, labels = data[:-1, :], data[-1, :]
    means = utils.fc_mean(data)
    pdata = v.T @ data
    x = pdata[0, :]
    y = pdata[1, :]

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.scatter(x[labels == 1], y[labels == 1], alpha=0.5, color=utils.green)
    plt.scatter(x[labels == 0], y[labels == 0], alpha=0.5, color=utils.red)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.title("2-D Projection (Original Data)")

    dmean = utils.fc_mean(data)
    dstd = data.std(axis=1).reshape((11, 1))
    norm = (data - dmean) / dstd

    w, v = utils.PCA(norm, feat_label=False)
    w, v = w[:2], v[:, :2]
    pnorm = v.T @ norm

    x = pnorm[0, :]
    y = pnorm[1, :]

    plt.subplot(2, 1, 2)
    plt.scatter(x[labels == 1], y[labels == 1], alpha=0.5, color=utils.green)
    plt.scatter(x[labels == 0], y[labels == 0], alpha=0.5, color=utils.red)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.title("2-D Projection (Normalized Data)")

    plt.savefig(fout, format='jpg')
    # plt.show()
