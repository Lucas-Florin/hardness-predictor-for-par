import argparse

# TODO: Remove unnecessary options
# TODO: Simplify usage through standard options
# TODO: Unify nomenclature


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
    parser.add_argument('--width', type=int, default=192,
                        help='width of an image')
    parser.add_argument('--full-attributes', action='store_true',
                        help='use all attributes available, not only the ones selected in the original paper')


    # ************************************************************
    # Optimization options
    # ************************************************************
    parser.add_argument('--optim', type=str, default='sgd',
                        help='optimization algorithm (see optimizers.py)')
    parser.add_argument('--lr', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--weight-decay', default=5e-04, type=float,
                        help='weight decay')
    parser.add_argument('--optim-group-pretrained', action="store_true",
                        help='group parameters by pretrained and fresh')
    # sgd
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum factor for SGD')

    # ************************************************************
    # Training hyperparameters
    # ************************************************************

    # Epoch schedule
    parser.add_argument('--max-epoch', default=-1, type=int,
                        help='maximum epochs to run training function')
    parser.add_argument('--main-net-train-epochs', default=-1, type=int,
                        help='maximum epochs to train the main-Net (not including finetuning)')
    parser.add_argument('--hp-net-train-epochs', default=-1, type=int,
                        help='maximum epochs to train the HP-Net')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful when restart)')
    parser.add_argument('--hp-epoch-offset', type=int, default=0,
                        help='delay start and learning rate decay of HP-Net')
    parser.add_argument('--main-net-finetuning-epochs', type=int, default=0,
                        help='after the HP-Net has finished training and hard examples have been discarded, continue '
                             'training the main-Net. ')

    # Batch size
    parser.add_argument('--train-batch-size', default=32, type=int,
                        help='training batch size')
    parser.add_argument('--test-batch-size', default=32, type=int,
                        help='test batch size')

    # Training Data
    parser.add_argument('--train-val', action='store_true',
                        help='use validation split for training too')

    # Loss Function
    parser.add_argument('--loss-func', type=str, default='deepmar', choices=['scel', 'sscel', 'deepmar'],
                        help='name of the desired loss function')
    parser.add_argument('--loss-func-param', type=float, default=1,
                        help='the parameter for the main loss function')

    # Realistic Predictor
    parser.add_argument('--no-hp-feedback', action='store_true',
                        help='do not use the hardness score as weighting for the main net loss function')
    parser.add_argument('--hp-train-sequentially', action='store_true',
                        help='train the HP-Net only after the main net is fully trained')
    parser.add_argument('--train-hp-only', action='store_true',
                        help='only train the HP-Net')
    parser.add_argument('--use-deepmar-for-hp', action='store_true',
                        help='use DeepMAR weighting for the HP loss')
    parser.add_argument('--hp-loss-param', type=float, default=1.0,
                        help='the parameter for the HP loss function')
    parser.add_argument('--use-bbs-gt', action='store_true',
                        help='use bounding boxes as ground truth to train the HP-Net')
    parser.add_argument('--use-bbs-feedback', action='store_true',
                        help='use bounding boxes for hardness score feedback')
    parser.add_argument('--hp-visibility-weight', type=float, default=1.0,
                        help='the weighting factor for the bounding boxes ground truth')


    # HP-Loss calibration
    parser.add_argument('--hp-calib', type=str, default='none', choices=['none', 'linear'],
                        help='calibrator for the HP-Loss function')
    parser.add_argument('--hp-calib-thr', type=str, default='f1', choices=['f1', 'mean'],
                        help='calibrator for the HP-Loss function')
    parser.add_argument('--f1-baseline', type=str, default='',
                        help='load baseline F1 calibration thresholds from previous model')


    # ************************************************************
    # Learning rate scheduler options
    # ************************************************************
    parser.add_argument('--lr-scheduler', type=str, default='multi_step',
                        help='learning rate scheduler (see lr_schedulers.py)')
    parser.add_argument('--stepsize', default=[15, 20, 25], nargs='+', type=int,
                        help='stepsize to decay learning rate')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='learning rate decay')
    parser.add_argument('--hp-net-lr-multiplier', default=1.0, type=float,
                        help='initial learning rate multiplier for the HP-Net w.r.t. the main model')

    # ************************************************************
    # Architecture
    # ************************************************************
    parser.add_argument('-m', '--model', type=str, default='resnet50')
    parser.add_argument('--hp-model', type=str, default='resnet50')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='do not use pretrained weights')
    parser.add_argument('--hp-net-simple', action='store_true',
                        help='predict hardness scores for entire pictures, not for specific attributes')

    # ************************************************************
    # Test settings
    # ************************************************************
    parser.add_argument('--load-weights', type=str, default='',
                        help='load pretrained weights but ignore layers that don\'t match in size')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate only')
    parser.add_argument('--save-checkpoint', action='store_true',
                        help='save resulting model after evaluation')
    parser.add_argument('--group-atts', action='store_true',
                        help='group binary attributes into non-binary ones')
    parser.add_argument('--use-raw-acc', action='store_true',
                        help='use raw accuracy instead of mean accuracy')
    parser.add_argument('--f1-calib', action='store_true',
                        help='use F1 calibration')
    parser.add_argument('--f1-calib-split', type=str, default='train', choices=['test', 'val', 'train'],
                        help='name of the desired split for determining the f1-calibration thresholds (train/test/val)')
    parser.add_argument('--eval-freq', type=int, default=-1,
                        help='evaluation frequency (set to -1 to test only in the end)')
    parser.add_argument('--start-eval', type=int, default=0,
                        help='start to evaluate after a specific epoch')
    parser.add_argument('--eval-split', type=str, default='test', choices=['test', 'val', 'train'],
                        help='name of the desired evaluation split (train/test/val)')
    parser.add_argument('--no-cache', action='store_true',
                        help='do not use cached output data')


    # Realistic Predictor
    parser.add_argument('--use-confidence', action='store_true',
                        help='use inverse confidence instead of hardness score. ')
    parser.add_argument('--ap-baseline', type=str, default='',
                        help='load baseline average precision from previous model')

    # Hard sample rejection
    parser.add_argument('--rejector', type=str, default='none',
                        choices=['none', 'macc', 'median', 'quantile', 'threshold', 'f1'],
                        help='name of the desired rejection strategy')
    parser.add_argument('--max-rejection-quantile', default=-1.0, type=float,
                        help='reject at most this portion of the hardest testing examples of each attribute')
    parser.add_argument('--rejection-threshold', default=-1.0, type=float,
                        help='reject testing examples that are harder than this threshold')
    parser.add_argument('--reject-hard-attributes-quantile', default=-1.0, type=float,
                        help='reject this portion of the hardest (mean hardness score) attributes (training dataset)')
    parser.add_argument('--reject-hard-attributes-threshold', default=1.0, type=float,
                        help='reject attributes that have a mean hardness score higher than this threshold')
    parser.add_argument('--rejector-thresholds-split', type=str, default='val', choices=['test', 'val', 'train'],
                        help='name of the desired split for determining the rejector thresholds (train/test/val)')

    # ************************************************************
    # Plot settings
    # ************************************************************

    # Show easy/hard images
    parser.add_argument('--select-atts', type=str, nargs="+", default=[],
                        help='select attributes for analysis')
    parser.add_argument('--num-save-hard', type=int, default=0,
                        help='number of hard images that are saved to collage')
    parser.add_argument('--num-save-easy', type=int, default=0,
                        help='number of easy images that are saved to collage')
    parser.add_argument('--show-pos-samples', action='store_true',
                        help='only show examples with positive ground truth')
    parser.add_argument('--show-neg-samples', action='store_true',
                        help='only show examples with negative ground truth')

    # Plot data
    parser.add_argument('--plot-epoch-loss', action='store_true',
                        help='plot loss over epochs')
    parser.add_argument('--plot-acc-hp', action='store_true',
                        help='plot a metric over hardness')
    parser.add_argument('--plot-metric', type=str, default='macc', choices=['macc', 'f1'],
                        help='use a specific metric for the plot')
    parser.add_argument('--plot-pos-hp', action='store_true',
                        help='plot positivity rate over hardness')
    parser.add_argument('--plot-pos-atts', action='store_true',
                        help='plot positivity ratio over attributes')
    parser.add_argument('--plot-hp-hist', action='store_true',
                        help='plot hardness score histogram')
    parser.add_argument('--plot-x-max', type=float, default=1,
                        help='x axis limit')

    # Example images
    parser.add_argument('--show-example-imgs', action='store_true',
                        help='show example images with labels')
    parser.add_argument('--show-label-examples', action='store_true',
                        help='show example images for a specific attribute')
    parser.add_argument('--save-plot', action='store_true',
                        help='save plots as TikZ')
    parser.add_argument('--menu', action='store_true',
                        help='input commands at runtime')

    parser.add_argument('--show-example-bbs', action='store_true',
                        help='show example images with bounding boxes')


    # ************************************************************
    # Miscs
    # ************************************************************
    parser.add_argument('--print-freq', type=int, default=500,
                        help='print frequency')
    parser.add_argument('--fix-seed', action='store_true',
                        help='fixed seed for reproducibility')
    parser.add_argument('--resume', type=str, default='', metavar='PATH',
                        help='resume from a checkpoint')
    #parser.add_argument('--save-log', type=str, default='log',
    #                    help='path to save log files')
    parser.add_argument('--save-experiment', type=str, default='./experiments/')
    parser.add_argument('--use-cpu', action='store_true',
                        help='use cpu')
    parser.add_argument('--gpu-devices', default='0', type=str,
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--use-avai-gpus', action='store_true',
                        help='use available gpus instead of specified devices (useful when using managed clusters)')
    parser.add_argument('--keep-best-model', action='store_true',
                        help='save the best model checkpoint separately from the latest checkpoint')


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
        'train_val': parsed_args.train_val,
        'full_attributes': parsed_args.full_attributes,
        'use_bbs': parsed_args.use_bbs_gt or parsed_args.use_bbs_feedback
    }


def optimizer_kwargs(parsed_args):
    """
    Build kwargs for optimizer in optimizers.py from
    the parsed command-line arguments.
    """
    return {
        'optim': parsed_args.optim,
        'lr': parsed_args.lr,
        'group_pretrained': parsed_args.optim_group_pretrained,
        'weight_decay': parsed_args.weight_decay,
        'momentum': parsed_args.momentum
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