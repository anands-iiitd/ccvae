import torch
import torch.nn as nn
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.transforms import Compose, ToTensor


def create_directories(dir_config):
    """Create various directories as per the `config`."""
    if not os.path.exists(dir_config.output_path):
        os.makedirs(dir_config.output_path)

    if not os.path.exists(dir_config.model_path):
        os.makedirs(dir_config.model_path)

    if not os.path.exists(os.path.join(dir_config.model_path, "checkpoints")):
        os.makedirs(os.path.join(dir_config.model_path, "checkpoints"))


def set_seeds(seed):
    """Set various seed values."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


### OLD UTILS ###

# Compose a transform configuration.
transform_config = Compose([ToTensor()])


def mse_loss(input, target):
    return torch.sum((input - target).pow(2)) / input.data.nelement()


def l1_loss(input, target):
    return torch.sum(torch.abs(input - target)) / input.data.nelement()


def reparameterize(training, mu, logvar):
    if training:
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)
    else:
        return mu


def weights_init(layer):
    if isinstance(layer, nn.Conv2d):
        layer.weight.data.normal_(0.0, 0.05)
        layer.bias.data.zero_()
    elif isinstance(layer, nn.BatchNorm2d):
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.zero_()
    elif isinstance(layer, nn.Linear):
        layer.weight.data.normal_(0.0, 0.05)
        layer.bias.data.zero_()


def imshow_grid(images, shape=[2, 8], name='default', save=False):
    """
    Plot images in a grid of a given shape.
    Initial code from: https://github.com/pumpikano/tf-dann/blob/master/utils.py
    """
    fig = plt.figure(1)
    grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.05)

    size = shape[0] * shape[1]
    for i in range(size):
        grid[i].axis('off')
        grid[i].imshow(images[i])  # The AxesGrid object work as a list of axes.

    if save:
        plt.savefig('reconstructed_images/' + str(name) + '.png')
        plt.clf()
    else:
        plt.show()
