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
import evaluation.metrics as metrics


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
                  attribute_name=None, labels=None, hardness=None, prediction_probs=None, predictions=None):
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
    if labels is not None and hardness is not None and prediction_probs is not None and predictions is not None:
        # Display label and hardness score for each image.
        for cell, l, prob, pred, h in zip(ax.flat, labels.flatten(), prediction_probs.flatten(),
                                          predictions.flatten(), hardness.flatten()):
            cell.title.set_text("{};{:.2f};{};{:.2f}".format(int(l), prob, int(pred), h))
    elif hardness is not None:
        # Display only hardness score for each image.
        for cell, h in zip(ax.flat, hardness.flatten()):
            cell.title.set_text("H:{0:.2f}".format(h))

    for cell in ax.flat:
        cell.set_axis_off()  # Turn off the axis. It is irrelevant here.


    plt.show()
    if input("Save Figure? (y/n):") == "y":
        plt.savefig(filename, format="png")
        print("Saved by hardness examples at " + filename)


def show_accuracy_by_hardness(filename, title, attribute_name, labels, predictions, hp_scores):
    x = np.arange(0, 1, 0.1)
    y = np.zeros(x.shape)
    num_datapoints = labels.shape[0]
    predictions = predictions.reshape((num_datapoints, 1))
    labels = labels.reshape((num_datapoints, 1))
    for i in range(len(x)):

        num_reject = int(num_datapoints * x[i])
        ignore = np.zeros(labels.shape, dtype="int8")
        sorted_idxs = hp_scores.argsort()
        hard_idxs = sorted_idxs[-num_reject:]
        if num_reject > 0:
            ignore[hard_idxs] = 1
        predictions = predictions.reshape((num_datapoints, 1))
        labels = labels.reshape((num_datapoints, 1))
        macc = metrics.mean_accuracy(predictions, labels, ignore)
        y[i] = macc
    fig, ax = plt.subplots()
    ax.plot(x, y)
    if attribute_name:
        title += "; Attribute = " + attribute_name
    fig.suptitle(title)
    plt.xlabel("Portion of rejected hard samples")
    plt.ylabel("Mean accuracy on remaining samples")
    #plt.ylim(0, 1)
    plt.show()
    if input("Save Figure? (y/n):") == "y":
        plt.savefig(filename, format="png")
        print("Saved by hardness examples at " + filename)


def show_positivity_by_hardness(filename, title, attribute_name, labels, predictions, hp_scores, resolution=10):
    x = np.arange(resolution)
    y = np.zeros(x.shape)
    num_datapoints = labels.shape[0]
    predictions = predictions.reshape((num_datapoints, 1))
    labels = labels.reshape((num_datapoints, 1))
    num_select = int(num_datapoints // resolution)
    num_rest = num_datapoints % resolution
    sorted_idxs = hp_scores.argsort()
    sorted_idxs = sorted_idxs[np.random.choice(sorted_idxs.shape[0], num_datapoints - num_rest, replace=False)]
    print("Ignoring {} randomly selected samples to avoid rest. ". format(num_rest))
    assert num_select * resolution == sorted_idxs.shape[0]
    for i in range(resolution):
        selected_idx = sorted_idxs[num_select * i:num_select * (i + 1)]
        y[i] = labels[selected_idx].sum() / num_select
    fig, ax = plt.subplots()
    ax.plot(x, y)
    if attribute_name:
        title += "; Attribute = " + attribute_name
    fig.suptitle(title)
    plt.xlabel("Quantile by Hardness")
    plt.ylabel("Positivity Rate")
    #plt.ylim(0, 1)
    plt.show()
    if input("Save Figure? (y/n):") == "y":
        plt.savefig(filename, format="png")
        print("Saved by hardness examples at " + filename)
