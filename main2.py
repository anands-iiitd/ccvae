"""Main script to run mean shift / k-means clustering algorithms."""

from warnings import filterwarnings
import utils
import config_loader

# Ignore warnings.
filterwarnings("ignore")

if __name__ == "__main__":
    # Initial set-up.
    config = config_loader.ConfigLoader("MNIST-mini")

    utils.set_seeds(config.train_config.seed)
    utils.create_directories(config.dir_config)

    # # Read data.
    # logging.info("Reading training data...")
    # full_data_path = os.path.join(config.DATA_DIR, config.dataset_name)
    # full_dataloaders = data.Dataloaders(config=config, data_path=full_data_path)
    #
    # logging.info("Reading test data...")
    # test_data_path = os.path.join(config.DATA_DIR, config.test_dataset_name)
    # test_dataloaders = data.Dataloaders(config=config, data_path=test_data_path)
    #
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
    #                     plot_type=PlotType.PRETRAINING, data_type=DataType.TEST)
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
