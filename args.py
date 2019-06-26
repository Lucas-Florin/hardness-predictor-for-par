import argparse

# TODO: Modularize


def argument_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ************************************************************
    # Datasets (general)
    # ************************************************************
    parser.add_argument('--root', type=str, default='datasets',
                        help='root path to data directory')
    parser.add_argument('-d', '--dataset-name', type=str, required=True,
                        help='name of the desired dataset')
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help='number of data loading workers (tips: 4 or 8 times number of gpus)')
    parser.add_argument('--height', type=int, default=256,
                        help='height of an image')
    parser.add_argument('--width', type=int, default=128,
                        help='width of an image')

    # ************************************************************
    # Data augmentation
    # ************************************************************
    parser.add_argument('--random-erase', action='store_true',
                        help='use random erasing for data augmentation')
    parser.add_argument('--color-jitter', action='store_true',
                        help='randomly change the brightness, contrast and saturation')
    parser.add_argument('--color-aug', action='store_true',
                        help='randomly alter the intensities of RGB channels')

    # ************************************************************
    # Optimization options
    # ************************************************************
    parser.add_argument('--optim', type=str, default='adam',
                        help='optimization algorithm (see optimizers.py)')
    parser.add_argument('--lr', default=0.0001, type=float,
                        help='initial learning rate')
    parser.add_argument('--weight-decay', default=5e-04, type=float,
                        help='weight decay')
    # sgd
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum factor for sgd and rmsprop')
    parser.add_argument('--sgd-dampening', default=0, type=float,
                        help='sgd\'s dampening for momentum')
    parser.add_argument('--sgd-nesterov', action='store_true',
                        help='whether to enable sgd\'s Nesterov momentum')
    # rmsprop
    parser.add_argument('--rmsprop-alpha', default=0.99, type=float,
                        help='rmsprop\'s smoothing constant')
    # adam/amsgrad
    parser.add_argument('--adam-beta1', default=0.9, type=float,
                        help='exponential decay rate for adam\'s first moment')
    parser.add_argument('--adam-beta2', default=0.999, type=float,
                        help='exponential decay rate for adam\'s second moment')

    # ************************************************************
    # Training hyperparameters
    # ************************************************************
    parser.add_argument('--max-epoch', default=60, type=int,
                        help='maximum epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful when restart)')

    parser.add_argument('--train-batch-size', default=32, type=int,
                        help='training batch size')
    parser.add_argument('--test-batch-size', default=100, type=int,
                        help='test batch size')

    parser.add_argument('--always-fixbase', action='store_true',
                        help='always fix base network and only train specified layers')
    parser.add_argument('--fixbase-epoch', type=int, default=0,
                        help='how many epochs to fix base network (only train randomly initialized classifier)')
    parser.add_argument('--open-layers', type=str, nargs='+', default=['classifier'],
                        help='open specified layers for training while keeping others frozen')

    parser.add_argument('--staged-lr', action='store_true',
                        help='set different lr to different layers')
    parser.add_argument('--new-layers', type=str, nargs='+', default=['classifier'],
                        help='newly added layers with default lr')
    parser.add_argument('--base-lr-mult', type=float, default=0.1,
                        help='learning rate multiplier for base layers')
    parser.add_argument('--loss-func', type=str, default='scel', choices=['scel', 'sscel', 'deepmar'],
                        help='name of the desired loss function')
    parser.add_argument('--loss-func-param', type=float, default=1,
                        help='the parameter for the loss function')


    # ************************************************************
    # Learning rate scheduler options
    # ************************************************************
    parser.add_argument('--lr-scheduler', type=str, default='multi_step',
                        help='learning rate scheduler (see lr_schedulers.py)')
    parser.add_argument('--stepsize', default=[20, 40], nargs='+', type=int,
                        help='stepsize to decay learning rate')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='learning rate decay')

    # ************************************************************
    # Architecture
    # ************************************************************
    parser.add_argument('-m', '--model', type=str, default='resnet50')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='do not load pretrained weights')
    parser.add_argument('--hp-net-simple', action='store_true',
                        help='predict hardness scores for entire pictures, not for specific attributes')

    # ************************************************************
    # Test settings
    # ************************************************************
    parser.add_argument('--load-weights', type=str, default='',
                        help='load pretrained weights but ignore layers that don\'t match in size')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate only')
    parser.add_argument('--group-atts', action='store_true',
                        help='group binary attributes into non-binary ones')
    parser.add_argument('--use-macc', action='store_true',
                        help='use mean accuracy instead of normal accuracy')
    parser.add_argument('--f1-calib', action='store_true',
                        help='use F1 calibration')
    parser.add_argument('--eval-freq', type=int, default=-1,
                        help='evaluation frequency (set to -1 to test only in the end)')
    parser.add_argument('--start-eval', type=int, default=0,
                        help='start to evaluate after a specific epoch')
    parser.add_argument('--eval-split', type=str, default='test', choices=['test', 'val'],
                        help='name of the desired evaluation split (test/val)')
    parser.add_argument('--num-save-hard', type=int, default=0,
                        help='number of hard images that are saved to collage')

    # ************************************************************
    # Miscs
    # ************************************************************
    parser.add_argument('--print-freq', type=int, default=100,
                        help='print frequency')
    parser.add_argument('--seed', type=int, default=1,
                        help='manual seed')
    parser.add_argument('--resume', type=str, default='', metavar='PATH',
                        help='resume from a checkpoint')
    #parser.add_argument('--save-log', type=str, default='log',
    #                    help='path to save log files')
    parser.add_argument('--save-experiment', type=str, default='experiments/experiment')
    parser.add_argument('--use-cpu', action='store_true',
                        help='use cpu')
    parser.add_argument('--gpu-devices', default='0', type=str,
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--use-avai-gpus', action='store_true',
                        help='use available gpus instead of specified devices (useful when using managed clusters)')
    parser.add_argument('--plot-epoch-loss', action='store_true',
                        help='plot loss over epochs')

    return parser


def image_dataset_kwargs(parsed_args):
    """
    Build kwargs for ImageDataManager in data_manager.py from
    the parsed command-line arguments.
    """
    return {
        'dataset_name': parsed_args.dataset_name,
        'root': parsed_args.root,
        'height': parsed_args.height,
        'width': parsed_args.width,
        'train_batch_size': parsed_args.train_batch_size,
        'test_batch_size': parsed_args.test_batch_size,
        'workers': parsed_args.workers,
        'random_erase': parsed_args.random_erase,
        'color_jitter': parsed_args.color_jitter,
        'color_aug': parsed_args.color_aug,
    }


def optimizer_kwargs(parsed_args):
    """
    Build kwargs for optimizer in optimizers.py from
    the parsed command-line arguments.
    """
    return {
        'optim': parsed_args.optim,
        'lr': parsed_args.lr,
        'weight_decay': parsed_args.weight_decay,
        'momentum': parsed_args.momentum,
        'sgd_dampening': parsed_args.sgd_dampening,
        'sgd_nesterov': parsed_args.sgd_nesterov,
        'rmsprop_alpha': parsed_args.rmsprop_alpha,
        'adam_beta1': parsed_args.adam_beta1,
        'adam_beta2': parsed_args.adam_beta2,
        'staged_lr': parsed_args.staged_lr,
        'new_layers': parsed_args.new_layers,
        'base_lr_mult': parsed_args.base_lr_mult,
    }


def lr_scheduler_kwargs(parsed_args):
    """
    Build kwargs for lr_scheduler in lr_schedulers.py from
    the parsed command-line arguments.
    """
    return {
        'lr_scheduler': parsed_args.lr_scheduler,
        'stepsize': parsed_args.stepsize,
        'gamma': parsed_args.gamma,
    }