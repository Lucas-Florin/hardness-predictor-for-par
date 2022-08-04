from torch.utils.data import DataLoader
import torch
import numpy as np

from .dataset_loader import ImageDataset
from .datasets import init_img_dataset
from .transforms import build_transforms


class BaseDataManager(object):

    def __init__(self,
                 device,
                 dataset_name,
                 root,
                 height,
                 width,
                 train_batch_size,
                 test_batch_size,
                 workers,
                 **kwargs
                 ):
        self.device = str(device)
        self.use_gpu = self.device != 'cpu'
        self.source_names = dataset_name
        self.root = root
        self.height = height
        self.width = width
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.workers = workers

        transform_train, transform_test = build_transforms(
            self.height, self.width, **kwargs
        )
        self.transform_train = transform_train
        self.transform_test = transform_test

    def return_dataloaders(self):
        """
        Return trainloader and testloader dictionary
        """
        return self.trainloader, self.testloader_dict

    def get_testloader(self, split):
        return self.testloader_dict[split]


class ImageDataManager(BaseDataManager):
    """
    Image data manager
    """

    def __init__(self,
                 device,
                 dataset_name,
                 train_val,
                 verbose=True,
                 **kwargs
                 ):
        super(ImageDataManager, self).__init__(device, dataset_name, **kwargs)
        dataset = init_img_dataset(name=dataset_name, verbose=verbose, **kwargs)
        self.dataset = dataset
        if verbose: print('=> Initializing TRAIN dataset')
        train = list()
        self.train = train
        # TODO: Change label dtype to long or bool
        for img_path, label in dataset.train:
                train.append((img_path, torch.tensor(label, dtype=torch.float)))
        if train_val:
            for img_path, label in dataset.val:
                train.append((img_path, torch.tensor(label, dtype=torch.float)))
        self.attributes = list(dataset.attributes)
        self.num_attributes = dataset.num_attributes
        # TODO: drop last?
        self.trainloader = DataLoader(
            ImageDataset(train, transform=self.transform_train),
            batch_size=self.train_batch_size, shuffle=True, num_workers=self.workers,
            pin_memory=self.use_gpu, pin_memory_device=self.device, drop_last=False
        )

        self.testloader_dict = dict()
        self.split_dict = dict()
        train_dataset = ImageDataset(train, transform=self.transform_test)
        self.testloader_dict['train'] = DataLoader(
            train_dataset,
            batch_size=self.test_batch_size, shuffle=False, num_workers=self.workers,
            pin_memory=self.use_gpu, pin_memory_device=self.device, drop_last=False
        )
        self.split_dict["train"] = train_dataset

        if verbose: print('=> Initializing TEST dataset')

        test = list()
        self.test = test
        for img_path, label in dataset.test:
            test.append((img_path, torch.tensor(label, dtype=torch.bool)))
        test_dataset = ImageDataset(test, transform=self.transform_test)
        self.testloader_dict['test'] = DataLoader(
            test_dataset,
            batch_size=self.test_batch_size, shuffle=False, num_workers=self.workers,
            pin_memory=self.use_gpu, pin_memory_device=self.device, drop_last=False
        )
        self.split_dict["test"] = test_dataset

        if verbose: print('=> Initializing VAL dataset')
        val = list()
        self.val = val
        for img_path, label in dataset.val:
            val.append((img_path, torch.tensor(label, dtype=torch.bool)))
        val_dataset = ImageDataset(val, transform=self.transform_test)
        self.testloader_dict['val'] = DataLoader(
            val_dataset,
            batch_size=self.test_batch_size, shuffle=False, num_workers=self.workers,
            pin_memory=self.use_gpu, pin_memory_device=self.device, drop_last=False
        )
        self.split_dict["val"] = val_dataset





