import torch
import torch.nn as nn


def init_optimizer(model,
                   optim='adam',  # optimizer choices
                   lr=0.003,  # learning rate
                   weight_decay=5e-4,  # weight decay
                   momentum=0.9,  # momentum factor for sgd and rmsprop
                   sgd_dampening=0,  # sgd's dampening for momentum
                   sgd_nesterov=False,  # whether to enable sgd's Nesterov momentum
                   rmsprop_alpha=0.99,  # rmsprop's smoothing constant
                   adam_beta1=0.9,  # exponential decay rate for adam's first moment
                   adam_beta2=0.999,  # # exponential decay rate for adam's second moment
                   ):

    param_groups = model.parameters()

    # Construct optimizer
    if optim == 'adam':
        return torch.optim.Adam(param_groups, lr=lr, weight_decay=weight_decay,
                                betas=(adam_beta1, adam_beta2))

    elif optim == 'amsgrad':
        return torch.optim.Adam(param_groups, lr=lr, weight_decay=weight_decay,
                                betas=(adam_beta1, adam_beta2), amsgrad=True)

    elif optim == 'sgd':
        return torch.optim.SGD(param_groups, lr=lr, momentum=momentum, weight_decay=weight_decay,
                               dampening=sgd_dampening, nesterov=sgd_nesterov)

    elif optim == 'rmsprop':
        return torch.optim.RMSprop(param_groups, lr=lr, momentum=momentum, weight_decay=weight_decay,
                                   alpha=rmsprop_alpha)

    else:
        raise ValueError('Unsupported optimizer: {}'.format(optim))