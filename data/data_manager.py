from torch.utils.data import DataLoader
import torch
import numpy as np

from .dataset_loader import ImageDataset, VideoDataset
from .datasets import init_img_dataset #, init_vid_dataset
from .transforms import build_transforms
from .samplers import build_train_sampler


class BaseDataManager(object):

    def __init__(self,
                 use_gpu,
                 source_names,
                 target_names,
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
        self.source_names = source_names
        self.target_names = target_names
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

    def return_testdataset_by_name(self, name):
        """
        Return query and gallery, each containing a list of (img_path, pid, camid).
        """
        return self.testdataset_dict[name]['query'], self.testdataset_dict[name]['gallery']


class ImageDataManager(BaseDataManager):
    """
    Image data manager
    """

    def __init__(self,
                 use_gpu,
                 source_names,
                 target_names,
                 **kwargs
                 ):
        super(ImageDataManager, self).__init__(use_gpu, source_names, target_names, **kwargs)

        print('=> Initializing TRAIN (source) datasets')
        train = list()
        for name in self.source_names:
            dataset = init_img_dataset(root=self.root, name=name)

            for img_path, label in dataset.train:
                train.append((img_path, torch.tensor(label.astype(np.float32))))
        self.attributes = dataset.attributes
        self.num_attributes = dataset.num_attributes
        self.trainloader = DataLoader(
            ImageDataset(train, transform=self.transform_train),
            batch_size=self.train_batch_size, shuffle=True, num_workers=self.workers,
            pin_memory=self.use_gpu, drop_last=True
        )

        print('=> Initializing TEST (target) datasets')
        self.testloader_dict = {name: {'test': None} for name in target_names}
        self.testdataset_dict = {name: {'test': None} for name in target_names}

        for name in self.target_names:
            dataset = init_img_dataset(
                root=self.root, name=name)

            test = list()
            for img_path, label in dataset.test:
                test.append((img_path, torch.tensor(label.astype(np.float32))))
            self.testloader_dict[name]['test'] = DataLoader(
                ImageDataset(test, transform=self.transform_test),
                batch_size=self.test_batch_size, shuffle=False, num_workers=self.workers,
                pin_memory=self.use_gpu, drop_last=False
            )

            self.testdataset_dict[name]['test'] = dataset.test

        print('\n')
        print('  **************** Summary ****************')
        print('  train names      : {}'.format(self.source_names))
        print('  # train datasets : {}'.format(len(self.source_names)))
        print('  # train images   : {}'.format(len(train)))
        print('  test names       : {}'.format(self.target_names))
        print('  *****************************************')
        print('\n')


class VideoDataManager(BaseDataManager):
    """
    Video-ReID data manager
    """

    def __init__(self,
                 use_gpu,
                 source_names,
                 target_names,
                 seq_len=15,
                 sample_method='evenly',
                 image_training=True,  # train the video-reid model with images rather than tracklets
                 **kwargs
                 ):
        super(VideoDataManager, self).__init__(use_gpu, source_names, target_names, **kwargs)
        self.seq_len = seq_len
        self.sample_method = sample_method
        self.image_training = image_training

        print('=> Initializing TRAIN (source) datasets')
        train = []
        self._num_train_pids = 0
        self._num_train_cams = 0

        for name in self.source_names:
            dataset = init_vidreid_dataset(root=self.root, name=name, split_id=self.split_id)

            for img_paths, pid, camid in dataset.train:
                pid += self._num_train_pids
                camid += self._num_train_cams
                if image_training:
                    # decompose tracklets into images
                    for img_path in img_paths:
                        train.append((img_path, pid, camid))
                else:
                    train.append((img_paths, pid, camid))

            self._num_train_pids += dataset.num_train_pids
            self._num_train_cams += dataset.num_train_cams

        self.train_sampler = build_train_sampler(
            train, self.train_sampler,
            train_batch_size=self.train_batch_size,
            num_instances=self.num_instances,
        )

        if image_training:
            # each batch has image data of shape (batch, channel, height, width)
            self.trainloader = DataLoader(
                ImageDataset(train, transform=self.transform_train), sampler=self.train_sampler,
                batch_size=self.train_batch_size, shuffle=False, num_workers=self.workers,
                pin_memory=self.use_gpu, drop_last=True
            )

        else:
            # each batch has image data of shape (batch, seq_len, channel, height, width)
            # note: this requires new training scripts
            self.trainloader = DataLoader(
                VideoDataset(train, seq_len=self.seq_len, sample_method=self.sample_method,
                             transform=self.transform_train),
                batch_size=self.train_batch_size, shuffle=True, num_workers=self.workers,
                pin_memory=self.use_gpu, drop_last=True
            )

        print('=> Initializing TEST (target) datasets')
        self.testloader_dict = {name: {'query': None, 'gallery': None} for name in target_names}
        self.testdataset_dict = {name: {'query': None, 'gallery': None} for name in target_names}

        for name in self.target_names:
            dataset = init_vidreid_dataset(root=self.root, name=name, split_id=self.split_id)

            self.testloader_dict[name]['query'] = DataLoader(
                VideoDataset(dataset.query, seq_len=self.seq_len, sample_method=self.sample_method,
                             transform=self.transform_test),
                batch_size=self.test_batch_size, shuffle=False, num_workers=self.workers,
                pin_memory=self.use_gpu, drop_last=False,
            )

            self.testloader_dict[name]['gallery'] = DataLoader(
                VideoDataset(dataset.gallery, seq_len=self.seq_len, sample_method=self.sample_method,
                             transform=self.transform_test),
                batch_size=self.test_batch_size, shuffle=False, num_workers=self.workers,
                pin_memory=self.use_gpu, drop_last=False,
            )

            self.testdataset_dict[name]['query'] = dataset.query
            self.testdataset_dict[name]['gallery'] = dataset.gallery

        print('\n')
        print('  **************** Summary ****************')
        print('  train names       : {}'.format(self.source_names))
        print('  # train datasets  : {}'.format(len(self.source_names)))
        print('  # train ids       : {}'.format(self.num_train_pids))
        if self.image_training:
            print('  # train images   : {}'.format(len(train)))
        else:
            print('  # train tracklets: {}'.format(len(train)))
        print('  # train cameras   : {}'.format(self.num_train_cams))
        print('  test names        : {}'.format(self.target_names))
        print('  *****************************************')


print('\n')