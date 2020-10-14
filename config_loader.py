"""Loads config for loading datasets and training / testing all models."""

import os
import torch
from torchvision import transforms


class ConfigLoader:
    """Defines and loads various config params."""

    class DirectoryConfigLoader:
        """Defines params for various directories."""

        def __init__(self, data_name):
            self.root_path = os.path.abspath('.')
            self.data_path = os.path.join(self.root_path, 'data/')
            self.output_path = os.path.join(self.root_path, 'output/')
            self.model_path = os.path.join(self.root_path, 'model/')
            self.data_name = data_name
            self.train_path = os.path.join(self.data_path,
                                           '{0}-train'.format(data_name))
            self.test_path = os.path.join(self.data_path,
                                          '{0}-test'.format(data_name))

    class TrainConfigLoader:
        """Defines params for training."""

        class NetworkConfigLoader:
            """Defines training network relalted params."""

            def __init__(self, data_name):
                self.latent_dim = 32
                self.encoder_params = {'num_layers': 3,
                                       'layer_type': 'conv2d',
                                       'kernel_size': [3, 5, 5],
                                       'stride': [2, 1, 1],
                                       'norm_type': 'batch',
                                       'num_initial_channels': 32,
                                       'padding': 1}
                self.decoder_params = {'num_layers': 3,
                                       'layer_type': 'convt2d',
                                       'kernel_size': [5, 5, 3],
                                       'stride': [1, 1, 2],
                                       'norm_type': 'batch',
                                       'num_initial_channels': 128,
                                       'padding': 1}
                # Dataset-specific training params.
                if data_name == "MNIST-mini" or data_name == "MNIST":
                    self.img_dim = 28
                    self.channels = 1
                    self.num_classes = 10
                    self.mean = (0.5,)
                    self.std = (0.5,)
                elif data_name == "YTF":
                    self.img_dim = 55
                    self.channels = 3
                    self.num_classes = 41
                    self.mean = (0.485, 0.456, 0.406)
                    self.std = (0.229, 0.224, 0.225)
                else:
                    raise ValueError("Unsupported data set: ", data_name)

                self.transforms = transforms.Compose(
                    [transforms.ToTensor(),
                     transforms.Normalize(mean=self.mean, std=self.std)])

        def __init__(self, data_name):
            self.network_config = self.NetworkConfigLoader(data_name)
            # Continue training from saved a point.
            self.resume_training = False
            self.test_frac = 0.1
            self.learning_rate = 0.0001
            self.dropout_prob = 0.1
            self.seed = 42
            self.optimizer = "adam"
            if data_name == "MNIST-mini":
                self.num_epochs = 5
                self.batch_size = 8
            else:
                self.num_epochs = 100
                self.batch_size = 64

            if torch.cuda.is_available():
                print("Using cuda:0 processor.")
                self.device = torch.device("cuda:0")
            else:
                print("Cuda not available, using cpu.")
                self.device = torch.device("cpu")

    class PlotConfigLoader():
        """Defines and loads various config params for plotting."""

        def __init__(self):
            """Initialize params to default values."""
            """
            dims: 2, 3
            color_palette: "hsv", "bright", "dark", "colorblind", "pastel"
            marker: 'o', 'v', '^', '<', '>', '8', 's', 'p', '*',
                            'h', 'H', 'D', 'd', 'P', 'X'
            show_legend: bool
            show_label: bool
            """
            self.tsne_config = {"dims": 2,
                                "verbose": 1,
                                "n_iter": 300,
                                "perplexity": 40}
            self.color_palette = "hsv"
            self.marker = ["o", "*", "-", "v"]
            self.show_legend = True
            self.xmin, self.xmax = -20, 20
            self.ymin, self.ymax = -20, 20
            self.zmin, self.zmax = -20, 20
            self.fig_height, self.fig_width = 16, 10
            self.legend_fontsize = 8
            self.xlabel_text = "x-axis"
            self.xlabel_fontsize = 15
            self.ylabel_text = "y-axis"
            self.ylabel_fontsize = 15
            self.zlabel_text = "z-axis"
            self.zlabel_fontsize = 15
            self.label_fontsize = 13
            self.show_label = True
            self.plot_tensorboard = True

    def __init__(self, data_name):
        self.dir_config = self.DirectoryConfigLoader(data_name)
        self.train_config = self.TrainConfigLoader(data_name)
        self.plot_config = self.PlotConfigLoader()
