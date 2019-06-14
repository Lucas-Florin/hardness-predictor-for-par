from torch.utils.data import DataLoader
import torch
import numpy as np

from .dataset_loader import ImageDataset
from .datasets import init_img_dataset
from .transforms import build_transforms


class BaseDataManager(object):

    def __init__(self,
                 use_gpu,
                 dataset_name,
                 root='data',
                 height=256,
                 width=128,
                 train_batch_size=32,
                 test_batch_size=100,
                 workers=4,
                 train_sampler='',
                 random_erase=False,  # use random erasing for data augmentation
                 color_jitter=False,  # randomly change the brightness, contrast and saturation
                 color_aug=False,  # randomly alter the intensities of RGB channels
                 **kwargs
                 ):
        self.use_gpu = use_gpu
        self.source_names = dataset_name
        self.root = root
        self.height = height
        self.width = width
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.workers = workers
        self.train_sampler = train_sampler
        self.random_erase = random_erase
        self.color_jitter = color_jitter
        self.color_aug = color_aug

        transform_train, transform_test = build_transforms(
            self.height, self.width, random_erase=self.random_erase, color_jitter=self.color_jitter,
            color_aug=self.color_aug
        )
        self.transform_train = transform_train
        self.transform_test = transform_test

    def return_dataloaders(self):
        """
        Return trainloader and testloader dictionary
        """
        return self.trainloader, self.testloader_dict


class ImageDataManager(BaseDataManager):
    """
    Image data manager
    """
    # TODO: Implement discarding hard training examples.
    # TODO: Test.

    def __init__(self,
                 use_gpu,
                 dataset_name,
                 **kwargs
                 ):
        super(ImageDataManager, self).__init__(use_gpu, dataset_name, **kwargs)

        print('=> Initializing TRAIN dataset')
        train = list()
        self.train = train

        dataset = init_img_dataset(root=self.root, name=dataset_name)
        self.dataset = dataset

        for img_path, label in dataset.train:
                train.append((img_path, torch.tensor(label.astype(np.float32))))
        self.attributes = dataset.attributes
        self.num_attributes = dataset.num_attributes
        self.trainloader = DataLoader(
            ImageDataset(train, transform=self.transform_train),
            batch_size=self.train_batch_size, shuffle=True, num_workers=self.workers,
            pin_memory=self.use_gpu, drop_last=True
        )

        self.testloader_dict = dict()

        self.testloader_dict['train'] = DataLoader(
            ImageDataset(train, transform=self.transform_test),
            batch_size=self.test_batch_size, shuffle=False, num_workers=self.workers,
            pin_memory=self.use_gpu, drop_last=False
        )

        print('=> Initializing TEST dataset')

        test = list()
        self.test = test
        for img_path, label in dataset.test:
            test.append((img_path, torch.tensor(label.astype(np.float32))))
        self.testloader_dict['test'] = DataLoader(
            ImageDataset(test, transform=self.transform_test),
            batch_size=self.test_batch_size, shuffle=False, num_workers=self.workers,
            pin_memory=self.use_gpu, drop_last=False
        )

        print('=> Initializing VAL dataset')
        val = list()
        self.val = val
        for img_path, label in dataset.val:
            val.append((img_path, torch.tensor(label.astype(np.float32))))
        self.testloader_dict['val'] = DataLoader(
            ImageDataset(val, transform=self.transform_test),
            batch_size=self.test_batch_size, shuffle=False, num_workers=self.workers,
            pin_memory=self.use_gpu, drop_last=False
        )

    def discard_examples(self, examples_to_discard, split="train"):
        if split == "train":
            split_data = self.train
        else:
            split_data = None

        split_data_new = [sample for sample, discard in zip(split_data, examples_to_discard) if not discard]

        self.testloader_dict[split] = DataLoader(
            ImageDataset(split_data_new, transform=self.transform_test),
            batch_size=self.test_batch_size, shuffle=False, num_workers=self.workers,
            pin_memory=self.use_gpu, drop_last=False
        )

    def reset_split(self, split="train"):
        if split == "train":
            split_data = self.train
        else:
            split_data = None

        self.testloader_dict[split] = DataLoader(
            ImageDataset(split_data, transform=self.transform_test),
            batch_size=self.test_batch_size, shuffle=False, num_workers=self.workers,
            pin_memory=self.use_gpu, drop_last=False
        )






