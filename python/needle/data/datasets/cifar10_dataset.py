import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        files = []
        if train:
            for i in range(1, 6):
                base_name = "/data_batch_" + str(i)
                files.append(base_folder + base_name)
        else:
            files.append(base_folder + "/test_batch")
        self.transforms = transforms
        self.p = p
        self.imgs,self.labels = self._unpickle(files)
        self.imgs = np.array([row.reshape(3,32,32) for row in self.imgs])
        print(self.labels.shape)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        imgs = self.imgs[index]
        label = self.labels[index]
        imgs = self.apply_transforms(imgs)
        return (imgs, label)
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return self.imgs.shape[0]
        ### END YOUR SOLUTION
    def _unpickle(self, files):
        import pickle
        imgs = []
        labels = []
        for file in files:
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
                imgs.append(dict[b'data'])
                labels.append(dict[b'labels'])
        return np.vstack(imgs), np.concatenate(labels)
