"""
@author Lucas Florin
@contact lucasflorin4@gmail.com
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.ticker as ticker
import tikzplotlib as tikz
import os.path as osp
from data.dataset_loader import read_image
import torchvision.utils as vutils
import torch
import evaluation.metrics as metrics
from evaluation.rejectors import QuantileRejector


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


def show_accuracy_over_hardness(filename, attribute_names, labels, predictions, hp_scores, metric="macc", save_plot=False):
    num_attributes = len(attribute_names)
    x = np.arange(0, 1, 0.01)
    y = np.zeros((x.size, num_attributes))
    sorted_idxs = hp_scores.argsort(0)
    sorted_scores = metrics.column_indexing(hp_scores, sorted_idxs)
    for i in range(len(x)):
        rejector = QuantileRejector(x[i])
        rejector.update_thresholds(labels, predictions, hp_scores, sorted_scores, verbose=False)
        ignore = np.logical_not(rejector(hp_scores))
        if metric == "macc":
            macc = metrics.mean_attribute_accuracies(predictions, labels, ignore)
        elif metric == "f1":
            raise NotImplementedError
        else:
            raise ValueError("invalid metric name")
        y[i, :] = macc
    fig, ax = plt.subplots()
    ax.plot(x, y)
    plt.xlabel("Portion of rejected hard samples")
    plt.ylabel("Mean accuracy on remaining samples")
    ax.legend(attribute_names, fancybox=True, framealpha=0)
    _format_ticks(ax)

    if save_plot:
        plt.savefig(filename + ".png", format="png")
        tikz.save(filename + ".tex")
        print("Saved positivity ratio by hardness at " + filename)
    plt.show()


def show_positivity_over_hardness(filename, attribute_names, labels, predictions, hp_scores, resolution=10, save_plot=False):
    num_datapoints = labels.shape[0]
    num_attributes = len(attribute_names)
    x = np.arange(0, 1, 1/resolution)
    y = np.zeros((x.size, num_attributes))
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
    plt.xlabel("Rejected quantile sorted by hardness")
    plt.ylabel("Positive label distribution")
    #plt.yscale("log")
    #plt.ylim(0, 1)
    ax.legend(attribute_names)
    if save_plot:
        plt.savefig(filename + ".png", format="png")
        tikz.save(filename + ".tex")
        print("Saved positivity ratio by hardness at " + filename)
    plt.show()


def plot_hardness_score_distribution(filename, attribute_names, hp_scores_test, hp_scores_train, save_plot=False,
                                     confidnece=False):
    num_datapoints = hp_scores_test.shape[0]
    num_attributes = len(attribute_names)
    xl = "Inverse confidence score" if confidnece else "Hardness score"
    """
    plt.hist(hp_scores, bins="auto", histtype="step")
    
    plt.xlabel(xl)
    plt.ylabel("Distribution")
    plt.legend(attribute_names)
    """
    fig, axs = plt.subplots(num_attributes, 1, sharex="all", sharey="all")
    for att in range(num_attributes):
        ax = axs[att]
        hp_scores_test_att = hp_scores_test[:, att]
        ax.hist(hp_scores_test_att, bins="auto", histtype="step")
        hp_scores_train_att = hp_scores_train[:, att]
        ax.hist(hp_scores_train_att, bins="auto", histtype="step")
        ax.set(xlim=(0, 1))
        ax.set_title(attribute_names[att], y=0.2)
    axs[-1].set(xlabel=xl, ylabel="Distribution")
    axs[-1].legend(["Validation", "Train"], fancybox=True, framealpha=0)

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


def _format_ticks(ax):
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    #ax.xaxis.set_ticks_position('top')
    #ax.yaxis.set_ticks_position('right')