from utils import load_train_data
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

        plt.bar(i-0.05, fmeanT, yerr=fvarT, width=0.1, color='blue')
        plt.bar(i+0.05, fmeanF, yerr=fvarF, width=0.1, color='red')

    plt.suptitle("(Normalized) Feature Mean and STD")
    redpatch = ptc.Patch(color="red", label="$Quality_{wine} < 6$")
    bluepatch = ptc.Patch(color="blue", label="$Quality_{wine} > 6$")
    plt.figlegend(handles=[redpatch, bluepatch],
                  handlelength=1, loc='upper right')
    plt.tight_layout()
    plt.xlabel("Feature #")
    plt.xticks(xaxis, xaxis+1)
    plt.yticks([], [])
    plt.savefig(fout, format='jpg')
    # plt.show()


def main():
    data = load_train_data()
    visualize(data)


if __name__ == '__main__':
    main()
