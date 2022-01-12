import torch
import torch.nn as nn


def init_optimizer(model,
                   optim,  # optimizer choices
                   lr,  # learning rate
                   group_pretrained,
                   weight_decay,  # weight decay
                   momentum,  # momentum factor for sgd and rmsprop
                   ):

    if group_pretrained:
        param_groups = [{'params': model.get_params_finetuning(), 'lr': lr * 0.1},
                        {'params': model.get_params_fresh(), 'lr': lr}]
    else:
        param_groups = model.parameters()

    # Construct optimizer
    if optim == 'adam':
        return torch.optim.Adam(param_groups, lr=lr, weight_decay=weight_decay)
    if optim == 'adamw':
        return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    elif optim == 'sgd':
        return torch.optim.SGD(param_groups, lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        raise ValueError('Unsupported optimizer: {}'.format(optim))