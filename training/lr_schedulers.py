import torch


def init_lr_scheduler(
    optimizer,
    lr_scheduler,  # learning rate scheduler
    stepsize,  # step size to decay learning rate
    gamma,  # learning rate decay
    lr,
    epochs,
    steps_per_epoch,
        ):
    if lr_scheduler == 'single_step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=stepsize[0], gamma=gamma)

    elif lr_scheduler == 'multi_step':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=stepsize, gamma=gamma)

    elif lr_scheduler == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=gamma, patience=4)

    elif lr_scheduler == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    elif lr_scheduler == '1cycle':
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            )

    else:
        raise ValueError('Unsupported lr_scheduler: {}'.format(lr_scheduler))

