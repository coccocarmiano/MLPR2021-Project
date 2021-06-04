from matplotlib import pyplot as plt
import matplotlib.patches as ptc
import numpy as np
from utils import load_train_data

fout = '../img/dist.jpg'
nfeatures = 12
nbins = 120


def visualize(data):
    selectorT = data[-1] == 1
    selectorF = data[-1] == 0

    plt.figure(figsize=(12, 8))
    for i in range(nfeatures-1):
        plt.subplot(4, 3, i+1)
        plt.hist(np.sort(data[i, selectorT]), stacked=True,
                  color="blue", alpha=.7, bins=nbins)
        plt.hist(np.sort(data[i, selectorF]), stacked=True,
                  color="red", alpha=.7, bins=nbins)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.title(f"Feature {i+1}")
    plt.subplot(4, 3, 12)
    plt.hist(np.sort(data[-1, selectorT]), color="blue")
    plt.hist(np.sort(data[-1, selectorF]), color="red")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.title("Label")
    plt.suptitle("Feature Distribution")
    redpatch = ptc.Patch(color="red", label="Good Wine")
    bluepatch = ptc.Patch(color="blue", label="Bad Wine")
    plt.figlegend(handles=[redpatch, bluepatch],
                  handlelength=1, loc='upper right')
    plt.tight_layout()
    plt.savefig(fout, format='jpg')
    # plt.show()


def main():
    data = load_train_data()
    visualize(data)


if __name__ == '__main__':
    main()
