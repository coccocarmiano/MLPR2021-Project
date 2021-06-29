from matplotlib import pyplot as plt
import numpy as np
import utils

fout = '../img/distn.jpg'

nfeatures = 12
nbins = 100

def visualize(data):
    selectorT = data[-1] == 1
    selectorF = data[-1] == 0
    data = utils.normalize(data)

    plt.figure(figsize=(12, 8))
    for i in range(nfeatures-1):
        plt.subplot(4, 3, i+1)
        plt.hist(np.sort(data[i, selectorT]), density=True,
                 color=utils.green, alpha=.7, bins=nbins)
        plt.hist(np.sort(data[i, selectorF]), density=True,
                 color=utils.red, alpha=.7, bins=nbins)
        plt.xticks([], [])
        #plt.yticks([], [])
        plt.title(f"Feature {i+1}")
    plt.subplot(4, 3, 12)
    plt.hist(np.sort(data[-1, selectorT]), color=utils.green)
    plt.hist(np.sort(data[-1, selectorF]), color=utils.red)
    plt.xticks([], [])
    #plt.yticks([], [])
    plt.title("Label")
    plt.suptitle("Feature Distribution (Normalized)")
    plt.figlegend(handles=utils.get_patches(),
                  handlelength=1, loc='upper right')
    plt.tight_layout()
    plt.savefig(fout, format='jpg')
    plt.show()


def main():
    data = utils.load_train_data()
    visualize(data)


if __name__ == '__main__':
    main()
