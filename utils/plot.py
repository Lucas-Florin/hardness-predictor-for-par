import numpy as np
import matplotlib.pyplot as plt
import os.path as osp


def plot_epoch_losses(epoch_losses, save_dir, ts=""):
    x = np.arange(1, 1 + epoch_losses.shape[0])
    y = epoch_losses
    fig, ax = plt.subplots()
    ax.plot(x, y)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training loss over epochs")

    fname = ts + "epoch_losses.png"
    fpath = osp.join(save_dir, fname)
    plt.savefig(fpath, format="png")
    print("Saved loss plot at " + fname)
    plt.show()