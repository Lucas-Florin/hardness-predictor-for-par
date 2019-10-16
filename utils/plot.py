"""
@author Lucas Florin
@contact lucasflorin4@gmail.com
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tikzplotlib as tikz
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

    if len(epoch_losses.shape) == 2 and epoch_losses.shape[1] == 2:
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
                  attribute_name=None, labels=None, hardness=None, prediction_probs=None, predictions=None, save_plot=False):
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

    if save_plot:
        plt.savefig(filename, format="png")
        print("Saved by hardness examples at " + filename)
    plt.show()


def show_accuracy_over_hardness(filename, title, attribute_names, labels, predictions, hp_scores, save_plot=False):
    x = np.arange(0, 1, 0.01)
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
    if attribute_names:
        title += "; Attribute = " + attribute_names
    fig.suptitle(title)
    plt.xlabel("Portion of rejected hard samples")
    plt.ylabel("Mean accuracy on remaining samples")
    ax.legend(attribute_names)
    #plt.ylim(0, 1)

    if save_plot:
        plt.savefig(filename, format="png")
        print("Saved accuracy over hardness at " + filename)
    plt.show()


def show_positivity_over_hardness(filename, attribute_names, labels, predictions, hp_scores, resolution=10, save_plot=False):
    num_datapoints = labels.shape[0]
    num_attributes = len(attribute_names)
    x = np.arange(0, 1, 1/resolution)
    y = np.zeros((x.size, num_attributes))
    #predictions = predictions.reshape((num_datapoints, 1))
    num_select = num_datapoints // resolution
    num_rest = num_datapoints % resolution
    sorted_idxs = hp_scores.argsort(0)
    sorted_idxs = sorted_idxs[np.random.choice(num_datapoints, num_datapoints - num_rest, replace=False), :]
    print("Ignoring {} randomly selected samples to avoid rest. ". format(num_rest))
    assert num_select * resolution == sorted_idxs.shape[0]
    for att_idx in range(num_attributes):
        att_idxs = sorted_idxs[:, att_idx]
        att_lables = labels[:, att_idx]
        for i in range(resolution):
            selected_idx = att_idxs[num_select * i:num_select * (i + 1)]
            y[i, att_idx] = att_lables[selected_idx].sum(0) / num_select
    fig, ax = plt.subplots()
    ax.plot(x, y)
    #if attribute_name:
    #    title += "; Attribute = " + attribute_name
    #fig.suptitle(title)
    plt.xlabel("Quantile sorted by hardness")
    plt.ylabel("Positive label distribution")
    #plt.yscale("log")
    #plt.ylim(0, 1)
    ax.legend(attribute_names)
    if save_plot:
        plt.savefig(filename + ".png", format="png")
        tikz.save(filename + ".tex")
        print("Saved positivity ratio by hardness at " + filename)
    plt.show()


def plot_positivity_ratio_over_attributes(attribute_names, positivity_ratios, filename, save_plot=False):
    fig, ax = plt.subplots()
    y_pos = np.arange(len(attribute_names)) + 1
    ax.bar(y_pos, positivity_ratios)
    plt.xticks(y_pos, attribute_names, rotation=90)
    plt.ylabel("Positivity ratio")
    plt.xlabel("Attributes")
    plt.tight_layout()
    if save_plot:
        plt.savefig(filename + ".png", format="png")
        tikz.save(filename + ".tex")
        print("Saved positivity ratio by hardness at " + filename)
    plt.show()

