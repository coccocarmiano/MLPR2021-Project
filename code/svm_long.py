import numpy as np
import utils
from classifiers import DualSVM_Score, DualSVM_Train

def polysvm(trd, ted):

    def get_polyfunction(c, d):

        def f(x1, x2):
            return (x1.T @ x2 + c) ** d
        return f

    tel = ted[-1, :]

    alphas = DualSVM_Train(trd, function=get_polyfunction(1, 3), factr=1., bound=1.)
    scores = DualSVM_Score(trd, alphas, ted, bias=1)        
    pred = scores > 1
    nc = (pred == tel).sum()
    nt = (len(pred))
    acc = nc/nt*100
    print(f"PolyAcc: {acc:.3f}")


def kernelsvm(trd, ted):

    def kernel(gamma, bias):

        def f(x1, x2):
            diff = x1-x2
            value = (diff * diff).sum() * gamma * -1
            value = np.exp(value)
            return value + bias

        return f

    tel = ted[-1, :]


    alphas = DualSVM_Train(trd, function=kernel(.1, 100), factr=1)
    scores = DualSVM_Score(trd, alphas, ted, bias=100)

    pred = scores > -42861.543
    nc = (pred == tel).sum()
    nt = (len(pred))
    acc = nc/nt*100
    print(f"Kernel: {acc:.3f}")



if __name__ == '__main__':
    dataset = utils.load_train_data()
    tedataset = utils.load_train_data()

    print("polysvm:")
    polysvm(dataset, tedataset)
    print("kernelsvm:")
    kernelsvm(dataset, tedataset)

    feats, labels = dataset[:-1, :], dataset[-1, :]
    tfeats, tlabels = tedataset[:-1, :], tedataset[-1, :]
    feats = utils.normalize(feats, other=tfeats)
    dataset = np.vstack((feats, labels))
    tedataset = np.vstack((tfeats, tlabels))

    print("polysvm (n):")
    polysvm(dataset, tedataset)
    print("kernelsvm (n):")
    kernelsvm(dataset, tedataset)
