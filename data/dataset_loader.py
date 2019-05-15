from PIL import Image
import os.path as osp

from torch.utils.data import Dataset


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    if not osp.exists(img_path):
        raise IOError('{} does not exist'.format(img_path))
    while True:
        try:
            img = Image.open(img_path).convert('RGB')
            break
        except IOError:
            print('IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'.format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person Attribute Recognition Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Get a datapoint from the dataset and load the image.
        :param index: the index of the desired datapoint.
        :return: the loaded image, the labels and the path to the image.
        """
        img_path, label = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, label, img_path
