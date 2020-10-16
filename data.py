"""Loads data sets and iterators, given a config."""

import os
import torch
import numpy as np
import pandas as pd
import random
from collections import namedtuple
from enum import Enum
from sklearn.preprocessing import LabelEncoder
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader


class DataType(Enum):
    TRAIN = 1
    TEST = 2


def image_from_npy(data_path, image_name, transforms):
    """Set various seed values."""
    image_path = os.path.join(data_path, image_name + '.npy')
    image = np.transpose(np.load(image_path), (1, 2, 0)).astype(np.uint8)
    if transforms is not None:
        image = transforms(image)
    return image


# A struct for storing the two same-class images and their label.
Data = namedtuple('Data', ['X1', 'X2', 'Y'])


class PairedDataset(Dataset):
    """Creates dataset objects for pair-wise data."""

    def __init__(self, data_path, csv_path, transforms):
        csv_data = pd.read_csv(csv_path)
        image_names = np.asarray(csv_data.iloc[:, 0])
        self.labels = torch.from_numpy(
            LabelEncoder().fit_transform(np.asarray(csv_data.iloc[:, 1])))
        self.images = []
        self.label_index_map = {}
        for i in range(image_names.size):
            self.images.append(image_from_npy(data_path=data_path,
                                              image_name=image_names[i],
                                              transforms=transforms))
            if not self.labels[i] in self.label_index_map:
                self.label_index_map[self.labels[i].item()] = []
            self.label_index_map[self.labels[i].item()].append(i)

    def __getitem__(self, index):
        """Return a same-label image pair."""
        rand_index = random.SystemRandom().choice(
            self.label_index_map[self.labels[index].item()])
        return self.images[index], self.images[rand_index], self.labels[index]

    def __len__(self):
        """Return length of the dataset."""
        return len(self.labels)


class PairedDataloader:
    """Creates a paired data loader."""

    def __init__(self, data_path, csv_path, transforms, batch_size, device):
        paired_dataset = PairedDataset(data_path=data_path,
                                       csv_path=csv_path,
                                       transforms=transforms)

        self.paired_dataloader = DataLoader(dataset=paired_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            drop_last=True)
        self.paired_iterator = iter(self.paired_dataloader)
        self.device = device

    def load_batch(self, loader, iterator):
        """Load a batch of data."""
        if len(loader) == 0:
            return Data(X1=[], X2=[], Y=[])
        try:
            image1, image2, label = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            image1, image2, label = next(iterator)
        # Use as_tensor to avoid copying.
        return Data(X1=torch.as_tensor(image1).float().to(self.device),
                    X2=torch.as_tensor(image2).float().to(self.device),
                    Y=label)

    def load_paired_batch(self):
        """Load a batch of paired data."""
        return self.load_batch(self.paired_dataloader, self.paired_iterator)
