import numpy as np
from utils import load_train_data, PCA
from matplotlib import pyplot as plt

fout = '../img/scatter.jpg'
nfeats = 11

if __name__ == '__main__':
    data = load_train_data()
    w, v = PCA(data, stats=True)
    w, v = w[:2], v[:, :2]

    data, labels = data[:-1, :], data[-1, :]
    means = data.mean(axis=1).reshape((nfeats, 1))
    cdata = data - means
    pdata = v.T.dot(cdata)
    x = pdata[0, :]
    y = pdata[1, :]

    plt.subplot(2, 1, 1)
    plt.scatter(x[labels == 1], y[labels == 1], alpha=0.3, color='blue')
    plt.scatter(x[labels == 0], y[labels == 0], alpha=0.3, color='red')
    plt.title("Scatter of the 2 Most Discriminant Features")

    w, v = w[:-1], v[:, -1]
    pdata = v.T.dot(cdata)
    plt.subplot(2, 1, 2)
    plt.hist(np.sort(pdata[labels==1]), bins=70, alpha=0.8, color='blue')
    plt.hist(np.sort(pdata[labels==0]), bins=70, alpha=0.8, color='red')
    plt.title("Histogram of the Most Discriminating Feature")
    # plt.show()
    # Todo: add legend
    plt.savefig(fout, format='jpg')
