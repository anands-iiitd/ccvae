"""Reads .h5 data and splits it into train / test using stratified sampling."""

import os
import h5py
import config_loader
from sklearn.model_selection import StratifiedShuffleSplit


def split_dataset(name, data_path, test_frac):
    """Split dataset with `name` into train and test data."""
    print("Processing dataset ", name, "...")
    train_dir = os.path.join(data_path, name + '-train/')
    test_dir = os.path.join(data_path, name + '-test/')

    # Create directory to save images, if it doesn't exist.
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
        print("Created Train Directory: ", train_dir)

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        print("Created Test Directory: ", test_dir)

    # Read hdf5 file.
    path = data_path + name + '/data4torch.h5'
    hf = h5py.File(path, 'r')
    data = hf['data'][:]
    labels = hf['labels'][:]

    dataset_split = StratifiedShuffleSplit(n_splits=1,
                                           test_size=test_frac,
                                           random_state=0)

    for train_index, test_index in dataset_split.split(data, labels):
        data_train, data_test = data[train_index], data[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]

    # Write train data in h5py format.
    train_path = train_dir + '/data4torch.h5'
    hf_train = h5py.File(train_path, 'w')
    hf_train.create_dataset('data', data=data_train)
    hf_train.create_dataset('labels', data=labels_train)

    # Write test data in h5py format.
    test_path = test_dir + '/data4torch.h5'
    hf_test = h5py.File(test_path, 'w')
    hf_test.create_dataset('data', data=data_test)
    hf_test.create_dataset('labels', data=labels_test)

    print("Completed processing ", name, "....")


dataset_names = ["MNIST",
                 "YTF"]

for dataset_name in dataset_names:
    config = config_loader.ConfigLoader(dataset_name)
    split_dataset(dataset_name, config.dir_config.data_path,
                  config.train_config.test_frac)
