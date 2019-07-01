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


def show_img_grid(dataset, idxs, filename, title=None,
                  attribute_name=None, labels=None, hardness=None):
    """
    Create a grid of specific images from a dataset. If parameters labels and hardness are passed, the
    label and hardness of each image are displayed above it.
    :param dataset: the ImageDataset from which the images are selected.
    :param idxs: The idxs in the dataset of the images to be selected.
    :param filename: The path to which to save the images.
    :param title: (Optional) Title for the figure.
    :param attribute_name: (Optional) The name of the attribute for which the hardness is analysed.
    :param labels: (Optional) An array of the ground truth labels for each image.
    :param hardness: (Optional) An array of the hardness scores for each image.
    :return:
    """
    batch = [(read_image(dataset[i][2])) for i in np.array(idxs).flatten()]
    num_imgs = len(batch)
    grid_height = 3
    grid_width = num_imgs // grid_height if num_imgs % grid_height == 0 else num_imgs // grid_height + 1
    # figsize=(20, 10) makes sure that the saved image is not too small.
    fig, ax = plt.subplots(grid_height, grid_width, figsize=(20, 10))
    if title is not None:
        if attribute_name:
            title += "; Attribute = " + attribute_name
        fig.suptitle(title)
    for cell, img in zip(ax.flat, batch):
        cell.imshow(img)
    if labels is not None and hardness is not None:
        # Display label and hardness score for each image.
        for cell, l, h in zip(ax.flat, labels.flatten(), hardness.flatten()):
            cell.title.set_text("{0};{1:.2f}".format(int(l), h))
    elif hardness is not None:
        # Display only hardness score for each image.
        for cell, h in zip(ax.flat, hardness.flatten()):
            cell.title.set_text("H:{0:.2f}".format(h))

    for cell in ax.flat:
        cell.set_axis_off()  # Turn off the axis. It is irrelevant here.

    plt.savefig(filename, format="png")
    print("Saved by hardness examples at " + filename)
    plt.show()

