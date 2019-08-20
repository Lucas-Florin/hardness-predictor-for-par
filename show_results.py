"""
@author Lucas Florin
@contact lucasflorin4@gmail.com
"""

import torch
import argparse
import numpy as np
import re
from utils.plot import plot_epoch_losses


def main():
    global args
    if torch.cuda.is_available() and not args.use_cpu:
        checkpoint = torch.load(args.load_path)
    else:
        checkpoint = torch.load(args.load_path, map_location='cpu')
    losses = checkpoint["losses"]
    ts = re.match(r".*/([\d\-_]+)checkpoint", args.load_path).group(1)
    plot_epoch_losses(losses, args.save_path, ts=ts)


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-l", "--load-path", type=str, required=True, help="path to the checkpoint. ")
    parser.add_argument("-s", "--save-path", type=str, default=None, help="path to save new plot. ")
    parser.add_argument("--use-cpu", action='store_true', help="path to save new plot. ")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main()

