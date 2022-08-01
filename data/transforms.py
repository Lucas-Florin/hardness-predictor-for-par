from PIL import Image
import random
import math

import torch
import torchvision.transforms as T


def build_transforms(height,
                     width,
                     randaugment=False,
                     randaugment_num=3,
                     randaugment_mag=4,
                     **kwargs
        ):
    # use imagenet mean and std as default
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    normalize = T.Normalize(mean=imagenet_mean, std=imagenet_std)

    # TODO: remove options from args or implement options here.
    if randaugment:
        # print(f'Using RandAugment with N={randaugment_num} and M={randaugment_mag} ')
        transform_train = T.Compose([
            T.Resize((height, width)),
            T.RandAugment(randaugment_num, randaugment_mag, num_magnitude_bins=10),
            T.ToTensor(),
            normalize
        ])
    else:
        transform_train = T.Compose([
            T.Resize((height, width)),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ])

    # build test transformations
    transform_test = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        normalize,
    ])
    return transform_train, transform_test
