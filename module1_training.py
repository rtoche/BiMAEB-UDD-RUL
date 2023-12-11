import json
import os.path
import torch
import argparse

import math
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import MSELoss

from Modules.Models import AutoEncoder
from Modules.functions import compute_recon_loss_on_training_data, get_normality_NASA_turbofan_data
from torch.utils.data import random_split


def run(args):
    print(args)
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define training hyper-parameters
    window_length = args.window_length
    normal_op_len = args.norm_op_len
    lr = args.lr
    split = args.split
    epochs = args.epochs
    batch_size = args.b_size
    metrics_dir = args.metrics_dir
    model_name = args.model_name
    dataset_name = args.dataset
    latent_size = args.l_size
    HPCC = args.HPCC

    # Get Datasets
    normalize = True
    norm_op_len_as_pct = True

    normality_dataset = get_normality_NASA_turbofan_data(dataset_name=dataset_name,
                                                         window_length=window_length,
                                                         normality_length=normal_op_len,
                                                         norm_op_len_as_pct=norm_op_len_as_pct,
                                                         normalize=normalize,
                                                         HPCC=HPCC)

    # Define module2 params
    number_ts_features = normality_dataset.number_ts_features

    # Define train and test split
    train_size = math.ceil(normality_dataset.size * split)
    test_size = normality_dataset.size - train_size
    train_dataset, test_dataset = random_split(normality_dataset, [train_size, test_size])

    # Define the DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    regularization = None
    reg_weight = 0.00
    loss_func = MSELoss()
    model = AutoEncoder(ts_number_features=number_ts_features,
                        latent_size=latent_size,
                        device=device,
                        model_name=model_name,
                        metrics_dir=metrics_dir,
                        regularization=regularization,
                        regularization_weight=reg_weight)

    # Define optimizer
    optimizer = Adam(model.parameters(), lr=lr)

    print()
    print("Window Length: {}".format(window_length))
    print("Normality Length: {}".format(normal_op_len))
    print("Training Samples: {}".format(train_size))
    print("Testing Samples: {}".format(test_size))
    print("Batch Size: {}".format(batch_size))
    print("Epochs: {}".format(epochs))
    print("Learning Rate: {}".format(lr))
    print("Regularization: {}".format(regularization))
    print("Split: {}".format(split))
    print("Latent size: {}".format(latent_size))
    print("Using Device: {}\n".format(device))
    print("Model: {}\n".format(model))

    # Train the module2
    model.fit(train_dataloader=train_dataloader, test_dataloader=test_dataloader,
              epochs=epochs, optimizer=optimizer, loss_function=loss_func, device=device)

    # Compute reconstruction error stats on training data
    recon_loss = MSELoss()
    training_errors_stats_dict = compute_recon_loss_on_training_data(model,
                                                                     model_type=model.model_type,
                                                                     loss_function=recon_loss,
                                                                     train_dataloader=train_dataloader,
                                                                     device=device)

    # Save Stats Dict
    training_errors_stats_json = "training_errors_stats.json"
    path_to_training_errors = os.path.join(metrics_dir, training_errors_stats_json)
    json_object = json.dumps(training_errors_stats_dict)
    with open(path_to_training_errors, 'w') as file:
        file.write(json_object)

    print()
    print("Training Reconstruction Error Statistics")
    print("====================================================")
    number_spaces_after = 25
    print("Average Recon. Error:".ljust(number_spaces_after) + str(training_errors_stats_dict["error_avg"]))
    print("1-Std. on Recon. Error:".ljust(number_spaces_after) + str(training_errors_stats_dict["error_std"]))
    print("Min Recon. Error:".ljust(number_spaces_after) + str(training_errors_stats_dict["error_min"]))
    print("Max Recon. Error:".ljust(number_spaces_after) + str(training_errors_stats_dict["error_max"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train with samples from time series with some window size")
    parser.add_argument("-model_name", required=True, type=str, help="Model name")
    parser.add_argument("-dataset", required=True, type=str, help="Dataset name")
    parser.add_argument("-window_length", default=1, type=int, help="Window length")
    parser.add_argument("-norm_op_len", default=30, type=int, help="Normal operational length as a percent in decimal.")
    parser.add_argument("-lr", default=0.0001, type=float, help="Learning rate")
    parser.add_argument("-split", default=0.70, type=float, help="Split percentage used for training")
    parser.add_argument("-epochs", default=2, type=int, help="Number training epochs")
    parser.add_argument("-b_size", default=512, type=int, help="Batch size")
    parser.add_argument("-l_size", type=int, default=16, help="Latent space size.")
    parser.add_argument("-metrics_dir", default="default_metrics_dir", type=str, help="Directory to store metrics")
    parser.add_argument("--HPCC", action="store_true", help="Variable specifying if we are on the HPCC.")
    parser.add_argument("--DEBUGGING", action="store_true", help="Create very small datasets for debugging purposes")

    file_arguments = parser.parse_args()

    run(args=file_arguments)
