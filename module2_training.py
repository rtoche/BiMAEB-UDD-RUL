import math
import torch
import argparse
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from Modules.Models import RULNeuralNetwork
from Modules.functions import get_rul_NASA_turbofan_data


def run(args):
    print(args)
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define training hyper-parameters
    normal_op_len = args.n
    lr = args.lr
    split = args.split
    epochs = args.epochs
    batch_size = args.b_size
    metrics_dir = args.metrics_dir
    model_name = args.model_name
    HPCC = args.HPCC
    dataset_name = args.dataset

    # Get Datasets
    normalize = True
    anomalies_only = True
    normal_op_len_as_pct = True
    dataset = get_rul_NASA_turbofan_data(dataset_name=dataset_name,
                                         normal_op_len=normal_op_len,
                                         normal_op_len_as_pct=normal_op_len_as_pct,
                                         normalize=normalize,
                                         anomalies_only=anomalies_only,
                                         HPCC=HPCC)
    # Define module2 params
    number_ts_features = dataset.number_ts_features

    # Define train and test split
    train_size = math.ceil(dataset.size * split)
    test_size = dataset.size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Define the DataLoaders
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = RULNeuralNetwork(ts_number_features=number_ts_features,
                             device=device,
                             model_name=model_name,
                             metrics_dir=metrics_dir)

    # Define optimizer
    optimizer = Adam(model.parameters(), lr=lr)
    loss_func = MSELoss()

    print()
    print("Training Samples: {}".format(train_size))
    print("Testing Samples: {}".format(test_size))
    print("Batch Size: {}".format(batch_size))
    print("Epochs: {}".format(epochs))
    print("Learning Rate: {}".format(lr))
    print("Split: {}".format(split))
    print("Using Device: {}\n".format(device))
    print("Model: {}\n".format(model))

    # Train the module2
    model.fit(train_dataloader=train_dataloader, test_dataloader=test_dataloader,
              epochs=epochs, optimizer=optimizer, loss_function=loss_func, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train module2 with samples from time series with some window size")
    parser.add_argument("-model_name", required=True, type=str, help="Model name")
    parser.add_argument("-dataset", required=True, type=str, help="Dataset name")
    parser.add_argument("-n", required=True, type=int, help="Normal operating length percent")
    parser.add_argument("-lr", default=0.001, type=float, help="Learning rate")
    parser.add_argument("-split", default=0.70, type=float, help="Split percentage used for training")
    parser.add_argument("-epochs", default=500, type=int, help="Number training epochs")
    parser.add_argument("-b_size", default=512, type=int, help="Batch size")
    parser.add_argument("-metrics_dir", default="default_metrics_dir", type=str, help="Directory to store metrics")
    parser.add_argument("--HPCC", action="store_true", help="Variable specifying if we are on the HPCC.")
    parser.add_argument("--DEBUGGING", action="store_true", help="Create very small datasets for debugging purposes")

    file_arguments = parser.parse_args()

    run(args=file_arguments)
