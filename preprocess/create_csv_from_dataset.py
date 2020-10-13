"""Preprocesses data to create CSV files with (image_name, label) format."""
import os
import csv
import h5py
import config_loader
import numpy as np


def create_csv_image_path(dataset_name, data_dir):
    """Create a 2-column CSV (name, label) for every image in dataset_name."""
    img_dir = os.path.join(data_dir, dataset_name + '_images/')

    # Create directory to save images, if it doesn't exist.
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
        print("Created directory: ", img_dir)

    # Read hdf5 file.
    path = data_dir + dataset_name + '/data4torch.h5'
    hf = h5py.File(path, 'r')
    data = hf['data'][:]
    labels = hf['labels'][:]

    # Header to insert in the CSV file.
    header = ['image_id', 'label']

    csv_path = os.path.join(data_dir, dataset_name + '_image_paths.csv')
    if os.path.exists(csv_path):
        print("Dataset", dataset_name, "already processed. Skipping...\n")
        return

    print("Processing dataset", dataset_name, "...")
    with open(csv_path, 'wt', newline='') as file:
        writer = csv.writer(file, delimiter=',')

        # Write the header first followed by the individual rows.
        writer.writerow(i for i in header)
        for i in range(len(labels)):
            y = labels[i]
            write_tocsv = [dataset_name + '_' + str(i), str(y)]
            writer.writerow(j for j in write_tocsv)

            # Save images in the respective image directories.
            x = data[i]
            np.save(img_dir + dataset_name + '_' + str(i) + ".npy", x)


dataset_names = ["MNIST-train",
                 "MNIST-test",
                 "MNIST",
                 "YTF-train",
                 "YTF-test",
                 "YTF",
                 "COIL-100-train",
                 "COIL-100-test",
                 "COIL-100",
                 "COIL-20-train",
                 "COIL-20-test",
                 "COIL-20",
                 "FRGC-train",
                 "FRGC-test",
                 "FRGC",
                 "USPS-train",
                 "USPS-test",
                 "USPS",
                 "CMU-PIE-train",
                 "CMU-PIE-test",
                 "CMU-PIE",
                 "TINY-IMAGENET",
                 "TINY-IMAGENET-mini"]

for dataset_name in dataset_names:
    data_dir = os.path.join(os.path.abspath('../../'), 'DATASET/')
    create_csv_image_path(dataset_name, data_dir)
