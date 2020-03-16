from __future__ import absolute_import

import sys
import os
import os.path as osp
import tabulate as tab

from .iotools import mkdir_if_missing


class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """  
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class AccLogger(object):
    """
    RankLogger records the accuracy obtained specified evaluation steps and provides a function
    to show the summarized results, which are convenient for analysis.
    """
    def __init__(self):
        self.epochs = []
        self.performance = []

    def write(self, epoch, performance):
        self.epochs.append(epoch)
        self.performance.append(performance)

    def get_data(self):
        return [self.epochs, self.performance]

    def show_summary(self):
        print('=> Performance Evolution')
        headers = ["Epoch", "Performance"]
        table = tab.tabulate(zip(self.epochs, self.performance), floatfmt='.2%', headers=headers)
        print(table)
