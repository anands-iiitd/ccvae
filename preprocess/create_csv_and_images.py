"""Preprocesses data to create CSV files with (image_name, label) format."""
import os
import csv
import h5py
import config_loader
import numpy as np


def create_csv_and_images(name, data_path):
    """Create:
     a) a 2-column CSV (name, label) for every image in name.
     b) directories with individual images."""
    img_dir = os.path.join(data_path, name + '_images/')

    # Create directory to save images, if it doesn't exist.
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
        print("Created directory: ", img_dir)

    # Read hdf5 file.
    path = data_path + name + '/data4torch.h5'
    hf = h5py.File(path, 'r')
    data = hf['data'][:]
    labels = hf['labels'][:]

    # Header to insert in the CSV file.
    header = ['image_id', 'label']

    csv_path = os.path.join(data_path, name + '_image_paths.csv')
    if os.path.exists(csv_path):
        print("Dataset", name, "already processed. Skipping...\n")
        return

    print("Processing dataset", name, "...")
    with open(csv_path, 'wt', newline='') as file:
        writer = csv.writer(file, delimiter=',')

        # Write the header first followed by the individual rows.
        writer.writerow(i for i in header)
        for i in range(len(labels)):
            y = labels[i]
            write_tocsv = [name + '_' + str(i), str(y)]
            writer.writerow(j for j in write_tocsv)

            # Save images in the respective image directories.
            x = data[i]
            np.save(img_dir + name + '_' + str(i) + ".npy", x)


dataset_names = ["MNIST-train",
                 "MNIST-test",
                 "MNIST",
                 "YTF-train",
                 "YTF-test",
                 "YTF"]

for dataset_name in dataset_names:
    dir_config = config_loader.ConfigLoader.DirectoryConfigLoader(dataset_name)
    create_csv_and_images(dataset_name, dir_config.data_path)
