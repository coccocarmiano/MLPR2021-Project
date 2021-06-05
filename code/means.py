from utils import load_train_data, get_patches, green, red
from matplotlib import pyplot as plt
import matplotlib.patches as ptc
import numpy as np

fout = '../img/means.jpg'
nfeatures = 12
nbins = 100


def visualize(data):
    selectorT = data[-1] == 1
    selectorF = data[-1] == 0
    xaxis = np.arange(11)

    plt.figure(figsize=(12, 8))
    for i in range(nfeatures-1):
        tdata = data[i]

        tmax = max(tdata)
        tmin = min(tdata)
        zdata = (tdata-tmin) / (tmax-tmin)

        fT = zdata[selectorT]
        fF = zdata[selectorF]

        fmeanT = fT.mean()
        fmeanF = fF.mean()

        fvarT = fT.std()
        fvarF = fF.std()

        plt.bar(i-0.05, fmeanT, yerr=fvarT, width=0.1, density=True, color=green)
        plt.bar(i+0.05, fmeanF, yerr=fvarF, width=0.1, density=True, color=red)

    plt.suptitle("(Normalized) Feature Mean and STD")
    plt.figlegend(handles=get_patches(),
                  handlelength=1, loc='upper right')
    plt.tight_layout()
    plt.xlabel("Feature #")
    plt.xticks(xaxis, xaxis+1)
    plt.yticks([], [])
    plt.savefig(fout, format='jpg')
    plt.show()


def main():
    data = load_train_data()
    visualize(data)


if __name__ == '__main__':
    main()
