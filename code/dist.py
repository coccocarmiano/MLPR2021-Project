from matplotlib import pyplot as plt
import numpy as np
from utils import load_train_data, get_patches, green, red

fout = '../img/dist.jpg'

nfeatures = 12
nbins = 100

def visualize(data):
    selectorT = data[-1] == 1
    selectorF = data[-1] == 0

    plt.figure(figsize=(12, 8))
    for i in range(nfeatures-1):
        plt.subplot(4, 3, i+1)
        plt.hist(np.sort(data[i, selectorT]), density=True,
                 color=green, alpha=.7, bins=nbins)
        plt.hist(np.sort(data[i, selectorF]), density=True,
                 color=red, alpha=.7, bins=nbins)
        plt.xticks([], [])
        #plt.yticks([], [])
        plt.title(f"Feature {i+1}")
    plt.subplot(4, 3, 12)
    plt.hist(data[-1, selectorT], color=green)
    plt.hist(data[-1, selectorF], color=red)
    plt.xticks([], [])
    #plt.yticks([], [])
    plt.title("Label")
    plt.suptitle("Feature Distribution")
    plt.figlegend(handles=get_patches(),
                  handlelength=1, loc='upper right')
    plt.tight_layout()
    plt.savefig(fout, format='jpg')
    #plt.show()


def main():
    data = load_train_data()
    visualize(data)


if __name__ == '__main__':
    main()
