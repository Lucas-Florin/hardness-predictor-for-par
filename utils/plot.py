"""
@author Lucas Florin
@contact lucasflorin4@gmail.com
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import os.path as osp
from data.dataset_loader import read_image
import torchvision.utils as vutils
import torch


def plot_epoch_losses(epoch_losses, save_dir=None, ts=None):
    x = np.arange(1, 1 + epoch_losses.shape[0])
    y = epoch_losses
    fig, ax = plt.subplots()
    ax.plot(x, y)

    legend = ["Main Net Loss"]

    if epoch_losses.shape[1] == 2:
        legend += ["HP Net Loss"]
    ax.legend(legend)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Training loss over epochs")

    if save_dir is not None:
        if ts is None:
            ts = time.strftime("%Y-%m-%d_%H-%M-%S_")
        fname = ts + "epoch_losses.png"
        fpath = osp.join(save_dir, fname)
        plt.savefig(fpath, format="png")
        print("Saved loss plot at " + fname)
    plt.show()


def save_img_collage(dataset, idxs, filename):
    batch = [dataset[i][0] for i in np.array(idxs).flatten()]
    vutils.save_image(batch, filename)

