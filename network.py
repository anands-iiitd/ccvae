"""Defines autoencoder model network for datasets."""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Encoder(nn.Module):
    """Defines the model for encoder."""

    def __init__(self, network_config, device):
        super().__init__()
        self.network_config = network_config
        self.device = device
        self.encoder_fc, self.final_features = self.make_layers()

        # Style embeddings
        self.style_mu_fc = nn.Linear(
            in_features=self.final_features,
            out_features=self.network_config.style_dim,
            bias=True)
        self.style_logvar_fc = nn.Linear(
            in_features=self.final_features,
            out_features=self.network_config.style_dim,
            bias=True)

        # Class embeddings
        self.class_fc = nn.Linear(
            in_features=self.final_features,
            out_features=self.network_config.class_dim,
            bias=True)

    def make_layers(self):
        """Define layers for the encoder."""
        layers = []
        encoder_params = self.network_config.encoder_params

        norm_type = encoder_params['norm_type']
        kernel_size = encoder_params['kernel_size']
        stride = encoder_params['stride']
        padding = encoder_params['padding']
        num_layers = encoder_params['num_layers']

        in_channels = self.network_config.channels
        out_channels = encoder_params['num_initial_channels']
        num_features = self.network_config.img_dim

        for i in range(num_layers):
            # Add convolution, normalization and ReLU layers.
            layers += [nn.Conv2d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=kernel_size[i],
                                 stride=stride[i],
                                 padding=padding[i])]
            if norm_type == 'batch':
                layers += [nn.BatchNorm2d(out_channels)]
            elif norm_type == 'instance':
                layers += [nn.InstanceNorm2d(num_features=out_channels,
                                             track_running_stats=True)]
            layers += [nn.LeakyReLU(),
                       nn.Dropout(p=self.network_config.dropout_prob)]

            num_features = np.int(
                (num_features - kernel_size[i] + 2 * padding) / stride[i]
                + 1)
            in_channels = out_channels
            out_channels = 2 * out_channels

        final_features = num_features * num_features * in_channels
        return nn.Sequential(*layers), final_features

    def forward(self, x):
        """Forward computations of the encoder."""
        x = self.encoder_fc(x)
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        return self.style_mu_fc(x).to(self.device), self.style_logvar_fc(x).to(
            self.device), self.class_fc(x).to(self.device)


class Decoder(nn.Module):
    """Defines the decoder."""

    def __init__(self, network_config, device, final_features):
        super().__init__()
        self.network_config = network_config
        self.device = device
        self.final_features = final_features
        self.style_fc = nn.Sequential(
            nn.Linear(in_features=self.network_config.style_dim,
                      out_features=self.final_features, bias=True),
            nn.LeakyReLU())
        self.class_fc = nn.Sequential(
            nn.Linear(in_features=self.network_config.class_dim,
                      out_features=self.final_features, bias=True),
            nn.LeakyReLU())
        self.decoder_fc = self.make_layers()

    def make_layers(self):
        """Define layers for the decoder."""
        layers = []
        decoder_params = self.network_config.decoder_params

        norm_type = decoder_params['norm_type']
        kernel_size = decoder_params['kernel_size']
        stride = decoder_params['stride']
        padding = decoder_params['padding']
        num_layers = decoder_params['num_layers']

        in_channels = decoder_params['num_initial_channels']
        for i in range(num_layers):
            if i < num_layers - 1:
                out_channels = in_channels / 2
            else:
                out_channels = self.network_config.channels

            # Add deconvolution, normalization and ReLU layers.
            layers += [nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size[i],
                stride=stride[i],
                padding=padding[i])]
            if norm_type == 'batch':
                layers += [nn.BatchNorm2d(out_channels)]
            elif norm_type == 'instance':
                layers += [nn.InstanceNorm2d(num_features=out_channels,
                                             track_running_stats=True)]
            layers += [nn.LeakyReLU(),
                       nn.Dropout(p=self.network_config.dropout_prob)]

            in_channels = in_channels * 2

        return nn.Sequential(*layers)

    def forward(self, style_embeddings, class_embeddings):
        """Forward computations of the decoder."""
        x = torch.cat(self.style_fc(style_embeddings),
                      self.class_fc(class_embeddings))
        x = self.decoder_fc(x)
        num_initial_channels = self.network_config.decoder_params[
            'num_initial_channels']
        feature_map_dim = np.int(
            np.sqrt(self.final_features / num_initial_channels))
        x = x.view(x.size(0), num_initial_channels, feature_map_dim,
                   feature_map_dim)
        return self.decoder_fc(x).to(self.device)


class Autoencoder(nn.Module):
    """Defines autoencoder architecture."""

    def __init__(self, network_config, device):
        super().__init__()
        self.network_config = network_config
        self.device = device
        self.encoder = Encoder(network_config=self.network_config,
                               device=self.device).to(self.device)
        self.decoder = Decoder(network_config=self.network_config,
                               device=self.device,
                               final_features=self.encoder.final_features).to(
            self.device)

    def forward(self, x, only_decode=False):
        """Forward computations of the autoencoder."""
        encoded = torch.zeros([], dtype=torch.float32)
        if not only_decode:
            encoded = self.encoder(x).to(self.device)
            decoded = self.decoder(encoded).to(self.device)
        else:
            decoded = self.decoder(x).to(self.device)

        return encoded, decoded
