from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from collections import OrderedDict

import torch
import torch.nn as nn


def adjust_learning_rate(optimizer, base_lr, epoch, stepsize=20, gamma=0.1,
                         linear_decay=False, final_lr=0, max_epoch=100):
    if linear_decay:
        # linearly decay learning rate from base_lr to final_lr
        frac_done = epoch / max_epoch
        lr = frac_done * final_lr + (1. - frac_done) * base_lr
    else:
        # decay learning rate by gamma for every stepsize
        lr = base_lr * (gamma ** (epoch // stepsize))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def set_bn_to_eval(m):
    # 1. no update for running mean and var
    # 2. scale and shift parameters are still trainable
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def open_all_layers(model):
    """
    Open all layers in model for training.

    Args:
    - model (nn.Module): neural net model.
    """
    model.train()
    for p in model.parameters():
        p.requires_grad = True


def open_specified_layers(model, open_layers):
    """
    Open specified layers in model for training while keeping
    other layers frozen.

    Args:
    - model (nn.Module): neural net model.
    - open_layers (list): list of layer names.
    """
    if isinstance(model, nn.DataParallel):
        model = model.module

    for layer in open_layers:
        assert hasattr(model, layer), '"{}" is not an attribute of the model, please provide the correct name'.format(layer)

    for name, module in model.named_children():
        if name in open_layers:
            module.train()
            for p in module.parameters():
                p.requires_grad = True
        else:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False


def count_num_param(model):
    num_param = sum(p.numel() for p in model.parameters()) / 1e+06

    if isinstance(model, nn.DataParallel):
        model = model.module

    if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Module):
        # we ignore the classifier because it is unused at test time
        num_param -= sum(p.numel() for p in model.classifier.parameters()) / 1e+06
    return num_param


def accuracy(output, target):
    """Computes the attribute accuracy"""
    with torch.no_grad():
        batch_size = target.size(0)
        num_attributes = target.size(1)
        num_targets = batch_size * num_attributes

        pred = output > 0.5
        correct = pred.int() == target.int()

        acc = correct.sum().float() / num_targets
        acc_atts = correct.sum(0).float() / batch_size
        return acc, acc_atts


def load_pretrained_weights(model, weight_path):
    """Load pretrianed weights to model

    Incompatible layers (unmatched in name or size) will be ignored

    Args:
    - model (nn.Module): network model, which must not be nn.DataParallel
    - weight_path (str): path to pretrained weights
    """
    checkpoint = torch.load(weight_path)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    if len(matched_layers) == 0:
        print('** ERROR: the pretrained weights "{}" cannot be loaded, please check the key names manually (ignored and continue)'.format(weight_path))
    else:
        print('Successfully loaded pretrained weights from "{}"'.format(weight_path))
        if len(discarded_layers) > 0:
            print("* The following layers are discarded due to unmatched keys or layer size: {}".format(discarded_layers))