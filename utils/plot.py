"""
@author Lucas Florin
@contact lucasflorin4@gmail.com
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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
    plt.title("Training loss over epochs for " + str(ts))

    if save_dir is not None:
        if ts is None:
            ts = time.strftime("%Y-%m-%d_%H-%M-%S_")
        fname = ts + "epoch_losses.png"
        fpath = osp.join(save_dir, fname)
        plt.savefig(fpath, format="png")
        print("Saved loss plot at " + fname)
    plt.show()


def save_img_collage(dataset, idxs, filename, title=None,
                     attribute_name=None, labels=None, hardness=None):
    batch = [(read_image(dataset[i][2])) for i in np.array(idxs).flatten()]
    num_imgs = len(batch)
    grid_height = 3
    grid_width = num_imgs // grid_height if num_imgs % grid_height == 0 else num_imgs // grid_height + 1
    fig, ax = plt.subplots(grid_height, grid_width, figsize=(20, 10))
    describe = False if labels is None or hardness is None or attribute_name is None else True
    if title is not None:
        if describe:
            title += "; Attribute = " + attribute_name
        fig.suptitle(title)
    for cell, img, l, h in zip(ax.flat, batch, labels.flatten(), hardness.flatten()):
        cell.imshow(img)
        if describe:
            cell.title.set_text("L:{0}; H:{1:.2f}".format(int(l), h))
    for cell in ax.flat:
        cell.set_axis_off()

    plt.savefig(filename, format="png")
    print("Saved by hardness examples at " + filename)
    plt.show()

