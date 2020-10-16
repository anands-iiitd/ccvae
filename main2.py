"""Main script to run mean shift / k-means clustering algorithms."""

from warnings import filterwarnings
import utils
import config_loader
import os
import data

# Ignore warnings.
filterwarnings("ignore")


def get_train_loader(dir_config, train_config):
    return data.PairedDataloader(
        data_path=dir_config.train_path,
        csv_path=dir_config.train_csv_path,
        transforms=train_config.network_config.transforms,
        batch_size=train_config.batch_size,
        device=train_config.device)


def get_test_loader(dir_config, train_config):
    return data.PairedDataloader(
        data_path=dir_config.test_path,
        csv_path=dir_config.test_csv_path,
        transforms=train_config.network_config.transforms,
        batch_size=train_config.batch_size,
        device=train_config.device)


if __name__ == "__main__":
    config = config_loader.ConfigLoader("MNIST-mini")
    utils.set_seeds(config.train_config.seed)
    utils.create_directories(config.dir_config)

    # Read data.
    train_loader = get_train_loader(config.dir_config, config.train_config)
    test_loader = get_train_loader(config.dir_config, config.train_config)


    # # Train the model.
    # logging.info("Initializing model...")
    # net = network.Autoencoder(config).to(config.device)
    #
    # logging.info("Building model...")
    # modeloperator = model.Trainer(config, net, full_dataloaders, plotter)
    #
    # logging.info("Plotting tSNE without training the model...")
    # plotter.plot_helper(dataloader=full_dataloaders.plot_dataloader, net=net,
    #                     plot_type=PlotType.RAW, data_type=DataType.TRAIN)
    # plotter.plot_helper(dataloader=test_dataloaders.plot_dataloader, net=net,
    #                     plot_type=PlotType.RAW, data_type=DataType.TEST)
    #
    # logging.info("Pre-training model...")
    # modeloperator.pretrain()
    #
    # logging.info("Plotting tSNE after pretraining the model...")
    # plotter.plot_helper(dataloader=full_dataloaders.plot_dataloader, net=net,
    #                     plot_type=PlotType.PRETRAINING,
    #                     data_type=DataType.TRAIN)
    # plotter.plot_helper(dataloader=test_dataloaders.plot_dataloader, net=net,
    #                     plot_type=PlotType.PRETRAINING,
    #                     data_type=DataType.TEST)
    #
    # logging.info("Training model...")
    # modeloperator.train(load_pretrained=True)
    #
    # logging.info("Plotting tSNE after training the model...")
    # plotter.plot_helper(dataloader=full_dataloaders.plot_dataloader, net=net,
    #                     plot_type=PlotType.TRAINING, data_type=DataType.TRAIN)
    # plotter.plot_helper(dataloader=test_dataloaders.plot_dataloader, net=net,
    #                     plot_type=PlotType.TRAINING, data_type=DataType.TEST)
    #
    # plotter.plot_images_helper(full_dataloaders)
