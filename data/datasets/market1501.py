import glob
import os.path as osp
import re
import numpy as np

from .base import BaseDataset


class Market1501(BaseDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'market'
    split_dir = 'data/datasets/splits'

    def __init__(self, root='data', verbose=True, market1501_500k=False, **kwargs):
        super(Market1501, self).__init__(root)
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        self.extra_gallery_dir = osp.join(self.dataset_dir, 'images')
        self.market1501_500k = market1501_500k
        self.train_label_file = osp.join(self.split_dir, 'market_train.csv')
        self.val_label_file = osp.join(self.split_dir, 'market_val.csv')
        self.test_label_file = osp.join(self.split_dir, 'market_test.csv')

        self._check_before_run()

        train, train_atts = self.load_from_file(self.train_label_file)
        val,val_atts = self.load_from_file(self.val_label_file)
        test, test_atts = self.load_from_file(self.test_label_file)
        if len(train_atts) != len(val_atts) != len(test_atts):
            raise Exception('Number of attributes doesn\'t match between dataset splits.')

        self.attributes = train_atts
        self.train = train
        self.val = val
        self.test = test

        self.num_attributes = len(self.attributes)
        self.train_imgs = len(train)
        self.val_imgs = len(val)
        self.test_imgs = len(test)

        if verbose:
            print('=> Market1501 loaded')
            self.print_dataset_statistics(train, val, test, train_atts)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError('"{}" is not available'.format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError('"{}" is not available'.format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError('"{}" is not available'.format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError('"{}" is not available'.format(self.gallery_dir))
        if self.market1501_500k and not osp.exists(self.extra_gallery_dir):
            raise RuntimeError('"{}" is not available'.format(self.extra_gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))
        return dataset

