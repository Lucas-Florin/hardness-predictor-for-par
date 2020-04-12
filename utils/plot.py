"""
@author Lucas Florin
@contact lucasflorin4@gmail.com
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import tikzplotlib as tikz
import os.path as osp
from data.dataset_loader import read_image
import evaluation.metrics as metrics
from evaluation.rejectors import QuantileRejector

# TODO: document
# TODO: remove unnecessary function parameters -> check with calling object
# TODO: Optional titles


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


def show_img_grid(dataset, idxs, filename, sample_labels=None, save_plot=False):
    """
    Create a grid of specific images from a dataset. If parameters labels and hardness are passed, the
    label and hardness of each image are displayed above it.
    :param dataset: the ImageDataset from which the images are selected.
    :param idxs: The idxs in the dataset of the images to be selected.
    :param filename: The path to which to save the images.
    :param title: (Optional) Title for the figure.
    :param attribute_name: (Optional) The name of the attribute for which the hardness is analysed.
    :param labels: (Optional) An array of the ground truth labels for each image.
    :param sample_labels: (Optional) An array of the labels to be shown above each image.
    :return:
    """
    batch = [(read_image(dataset[i][2])) for i in np.array(idxs).flatten()]
    num_imgs = len(batch)
    grid_height = 4
    grid_width = num_imgs // grid_height if num_imgs % grid_height == 0 else num_imgs // grid_height + 1
    fig, ax = plt.subplots(grid_height, grid_width)#, figsize=(20, 10))
    for cell, img in zip(ax.flat, batch):
        cell.imshow(img)
    """
    if labels is not None and hardness is not None and prediction_probs is not None and predictions is not None:
        # Display label and hardness score for each image.
        for cell, l, prob, pred, h in zip(ax.flat, labels.flatten(), prediction_probs.flatten(),
                                          predictions.flatten(), hardness.flatten()):
            cell.title.set_text("{};{:.2f};{};{:.2f}".format(int(l), prob, int(pred), h))
    """
    if sample_labels is not None:
        # Display only hardness score for each image.
        for cell, h in zip(ax.flat, sample_labels.flatten()):
            cell.title.set_text("{0:.2f}".format(h))

    for cell in ax.flat:
        cell.set_axis_off()  # Turn off the axis. It is irrelevant here.

    if save_plot:
        plt.savefig(filename, format="png")
        print("Saved by hardness examples at " + filename)
    plt.show()


def show_example_imgs(dataset, filename, save_plot=False, num_imgs=2):
    """
    Show example images from dataset.
    :param dataset: the ImageDataset from which the images are selected.
    :param filename: The path to which to save the images.
    :return:
    """
    attribute_names = dataset.attributes
    labels = dataset.labels.astype("bool")
    filenames = dataset.filenames
    dataset_size = labels.shape[0]
    while True:
        idxs = np.random.choice(np.arange(dataset_size), size=num_imgs)
        batch = [(read_image(filenames[i])) for i in np.array(idxs).flatten()]
        selected_labels = [attribute_names[labels[idxs[i], :]] for i in range(num_imgs)]
        selected_labels_strings = []
        for label_list in selected_labels:
            s = ""
            for l in label_list:
                s += l + "\n"
            selected_labels_strings.append(s)
        grid_height = 1
        grid_width = num_imgs * 2

        fig, ax = plt.subplots(grid_height, grid_width)
        for i in range(num_imgs):
            img_cell = ax.flat[i * 2]
            label_cell = ax.flat[i * 2 + 1]
            img_cell.imshow(batch[i])
            label_cell.text(0, 0.25, selected_labels_strings[i])

        for cell in ax.flat:
            cell.set_axis_off()  # Turn off the axis. It is irrelevant here.

        if save_plot:
            plt.savefig(filename, format="png")
            print("Saved by hardness examples at " + filename)
        plt.show()


def show_example_bbs(dataset, filename, save_plot=False, num_imgs=1):
    """
    Create a grid of specific images from a dataset. If parameters labels and hardness are passed, the
    label and hardness of each image are displayed above it.
    :param dataset: the ImageDataset from which the images are selected.
    :param filename: The path to which to save the images.
    :return:
    """
    labels = dataset.labels
    bb_coordinate_idxs = dataset.bb_coordinate_idxs
    origin_coordinate_idx = dataset.origin_coordinate_idx
    filenames = dataset.filenames
    dataset_size = labels.shape[0]
    while True:
        idx = np.random.randint(0, dataset_size)
        print("Image index: {}".format(idx))
        img = read_image(filenames[idx])

        fig, ax = plt.subplots(1)
        img_labels = labels[idx, :]

        ax.imshow(img)
        origin_coordinates = [img_labels[origin_coordinate_idx + 0], img_labels[origin_coordinate_idx + 1]]
        bb_coordinates = [[img_labels[l + 0] - origin_coordinates[0],
                           img_labels[l + 1] - origin_coordinates[1],
                           img_labels[l + 2],
                           img_labels[l + 3]]
                          for l in bb_coordinate_idxs]

        print(np.array(bb_coordinates))

        for c in bb_coordinates:
            bb = patches.Rectangle((c[0], c[1]), c[2], c[3], linewidth=4, edgecolor="r", facecolor="none")
            ax.add_patch(bb)

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


def show_positivity_over_hardness(filename, attribute_names, labels, predictions, hp_scores, resolution=50, save_plot=False):
    num_datapoints = labels.shape[0]
    num_attributes = len(attribute_names)
    x = np.arange(0, 1, 1/resolution)
    y = np.zeros((x.size, num_attributes))
    num_select = num_datapoints // resolution
    num_rest = num_datapoints % resolution
    sorted_idxs = hp_scores.argsort(0)
    randomly_selected_idx = np.random.choice(num_datapoints, num_datapoints - num_rest, replace=False)
    randomly_selected_idx.sort()
    print("Ignoring {} randomly selected samples to avoid rest. ". format(num_rest))
    assert num_select * resolution == randomly_selected_idx.size
    for att_idx in range(num_attributes):
        att_idxs = sorted_idxs[:, att_idx][randomly_selected_idx]
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


def plot_hardness_score_distribution(filename, attribute_names, hp_scores_train, hp_scores_val, hp_scores_test=None, x_max=1, save_plot=False,
                                     confidnece=False):
    num_datapoints = hp_scores_val.shape[0]
    num_attributes = len(attribute_names)
    xl = "Uncertainty score" if confidnece else "Hardness score"
    """
    plt.hist(hp_scores, bins="auto", histtype="step")
    
    plt.xlabel(xl)
    plt.ylabel("Distribution")
    plt.legend(attribute_names)
    """
    fig, axs = plt.subplots(num_attributes, 1, sharex="all", sharey="all")
    for att in range(num_attributes):
        ax = axs[att]

        hp_scores_train_att = hp_scores_train[:, att]
        y_avg, x_avg = get_plot_hist(hp_scores_train_att)
        ax.plot(x_avg, y_avg)
        #ax.hist(hp_scores_train_att, bins="auto", histtype="step")

        hp_scores_val_att = hp_scores_val[:, att]
        y_avg, x_avg = get_plot_hist(hp_scores_val_att)
        ax.plot(x_avg, y_avg)
        #ax.hist(hp_scores_val_att, bins="auto", histtype="step")
        if hp_scores_test is not None:
            hp_scores_test_att = hp_scores_test[:, att]
            y_avg, x_avg = get_plot_hist(hp_scores_test_att)
            ax.plot(x_avg, y_avg)
            #ax.hist(hp_scores_test_att, bins="auto", histtype="step")
        ax.set(xlim=(0, x_max))
        ax.set_title(attribute_names[att], y=0.2)
    axs[0].set(ylabel="Distribution")
    axs[-1].set(xlabel=xl)
    axs[0].legend(["Train", "Validation", "Test"], fancybox=True, framealpha=0)

    if save_plot:
        plt.savefig(filename + ".png", format="png")
        tikz.save(filename + ".tex")
        print("Saved positivity ratio by hardness at " + filename)
    plt.show()

def get_plot_hist(scores):
    y_val, x_val = np.histogram(scores, bins="auto")
    x_val_avg = [x_val[0]] + [(x_val[i] + x_val[i + 1]) / 2 for i in range(len(y_val))]
    y_val_avg = [0] + y_val.tolist()
    if x_val_avg[0] != 0:
        x_val_avg = [0] + x_val_avg
        y_val_avg = [0] + y_val_avg
    return y_val_avg, x_val_avg


def plot_positivity_ratio_over_attributes(attribute_names, positivity_ratios, filename, save_plot=False):
    idxs = (- positivity_ratios).argsort()
    fig, ax = plt.subplots()
    y_pos = np.arange(len(attribute_names)) + 1
    ax.bar(y_pos, positivity_ratios[idxs])

    attribute_names = np.array(attribute_names)
    plt.xticks(y_pos, attribute_names[idxs], rotation=90)
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