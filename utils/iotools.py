from __future__ import absolute_import

import os
import os.path as osp
import errno
import json
import shutil
from collections import OrderedDict

import torch

# TODO: Remove unneccesary parts


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def check_isfile(path):
    isfile = osp.isfile(path)
    if not isfile:
        print('=> Warning: no file found at "{}" (ignored)'.format(path))
    return isfile


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def save_checkpoint(state, fpath='checkpoint.pth.tar', is_best=False):
    # TODO: Restructure, couple with result_dict
    if len(osp.dirname(fpath)) != 0:
        mkdir_if_missing(osp.dirname(fpath))
    # remove 'module.' in state_dict's keys if necessary
    state_dicts = state['state_dicts']
    for i in range(len(state_dicts)):
        state_dict = state_dicts[i]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict[k] = v
        state_dicts[i] = new_state_dict
    # save
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_model.pth.tar'))