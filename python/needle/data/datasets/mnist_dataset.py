from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import struct
import gzip

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        self.imgs = self._parse_img(image_filename)
        self.labels = self._parse_label(label_filename)
        assert self.imgs.shape[0] == self.imgs.shape[0], "img size must equal to label size"
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        imgs = self.imgs[index]
        label = self.labels[index]
        if len(imgs.shape) > 1:
            imgs = np.vstack([self.apply_transforms(img.reshape(28, 28, 1)).flatten() for img in imgs])
        else:
            imgs = self.apply_transforms(imgs.reshape((28, 28, 1))).flatten()
        return (imgs, label)
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.labels.shape[0]
        ### END YOUR SOLUTION
    def _parse_label(self, filename):
        with gzip.open(filename, 'rb') as file:
            label_data = file.read()
            _, item_num = struct.unpack('>2i', label_data[:8])
            labels = struct.unpack(f'>{item_num}B', label_data[8:])
            return np.array(labels, dtype=np.uint8)
    
    def _parse_img(self, filename):
        with gzip.open(filename, 'rb') as file:
            image_data = file.read()
            _, num, row, col = struct.unpack('>iiii', image_data[:16])
            X = np.zeros((num, row * col), dtype=np.float32)
            for i in range(0, num):
                image_size = row * col
                data = struct.unpack(f'>{row * col}B', image_data[(16 + i * image_size):(16 + (i + 1) * image_size)])
                image = np.array(data, dtype=np.float32)
                X[i] = image / 255
            return X