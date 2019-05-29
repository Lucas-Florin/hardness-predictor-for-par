"""
@author Lucas Florin
@contact lucasflorin4@gmail.com
"""

import torch
import argparse
import numpy as np
from utils.plot import plot_epoch_losses


def main():
    global args
    if torch.cuda.is_available():
        checkpoint = torch.load(args.load_path)
    else:
        checkpoint = torch.load(args.load_path, map_location='cpu')
    losses = checkpoint["losses"]
    plot_epoch_losses(losses, args.save_path)


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-l", "--load-path", type=str, required=True, help="path to the checkpoint. ")
    parser.add_argument("-s", "--save-path", type=str, default=None, help="path to save new plot. ")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main()

