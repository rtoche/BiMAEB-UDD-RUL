import torch
import numpy as np
from torch.utils.data import DataLoader
from Modules.DatasetClasses import NormalityTurbofanDataset
from Modules.DatasetClasses import RULTurbofanDataset
from Modules.DatasetClasses import WindowRULTurbofanDataset


def get_window_size_rul_NASA_turbofan_data(sequence_length: int, normalize: bool, anomalies_only: bool, HPCC: bool):
    if HPCC:
        path_to_data_dir = "/home/rtoche/projects/DART-LP2-NASA/src/data/NASA_Turbofan"
    else:
        path_to_data_dir = "/Users/rafaeltoche/Documents/School/Research/" \
                           "Rainwaters_Lab/DART-LP2/Condition_Monitoring/NASA_turbofan_data/train"

    dataset_csv_file_name = "FD001_train_unsupervised_labels_VariationalAutoEncoder.csv"

    print(f"Loading {dataset_csv_file_name}...")
    dataset = WindowRULTurbofanDataset(data_dir_path=path_to_data_dir,
                                       dataset_csv_name=dataset_csv_file_name,
                                       sequence_length=sequence_length,
                                       normalize=normalize,
                                       anomalies_only=anomalies_only,
                                       DEBUGGING=False)
    return dataset


def get_rul_NASA_turbofan_data(dataset_name: str,
                               normal_op_len: int,
                               normal_op_len_as_pct: bool,
                               normalize: bool,
                               anomalies_only: bool,
                               HPCC: bool):
    if HPCC:
        path_to_data_dir = "/home/rtoche/projects/DART-LP2-NASA/src/data/NASA_Turbofan"
    else:
        path_to_data_dir = "/Users/rafaeltoche/Documents/School/Research/" \
                           "Rainwaters_Lab/DART-LP2/Condition_Monitoring/data/NASA_turbofan_data/train"

    prefix = "Pct" if normal_op_len_as_pct else "Len"
    dataset_csv_file_name = f"{dataset_name}_train_unsupervised_labels_AutoEncoder_Op{prefix}{normal_op_len}.csv"

    print(f"Loading {dataset_csv_file_name}...")
    dataset = RULTurbofanDataset(data_dir_path=path_to_data_dir,
                                 dataset_csv_name=dataset_csv_file_name,
                                 normalize=normalize,
                                 anomalies_only=anomalies_only,
                                 DEBUGGING=False)
    return dataset


def get_normality_NASA_turbofan_data(dataset_name: str,
                                     window_length: int,
                                     normality_length: int,
                                     norm_op_len_as_pct: bool,
                                     normalize: bool, HPCC: bool):
    if HPCC:
        path_to_data_dir = "/home/rtoche/projects/DART-LP2-NASA/src/data/NASA_Turbofan"
    else:
        path_to_data_dir = "/Users/rafaeltoche/Documents/School/Research/Rainwaters_Lab/DART-LP2/Condition_Monitoring/data/NASA_turbofan_data/train/"

    dataset_csv_file_name = f"{dataset_name}_train.csv"

    data = NormalityTurbofanDataset(data_dir_path=path_to_data_dir,
                                                      dataset_csv_name=dataset_csv_file_name,
                                                      window_length=window_length,
                                                      norm_op_len=normality_length,
                                                      norm_op_as_pct=norm_op_len_as_pct,
                                                      normalize=normalize,
                                                      DEBUGGING=False)

    return data


def compute_recon_loss_on_training_data(model,
                                        model_type: str,
                                        loss_function,
                                        train_dataloader: DataLoader,
                                        device: str):

    # Compute the average loss for training data
    training_losses = []
    with torch.no_grad():
        model.eval()
        for batch in train_dataloader:
            features, targets = batch

            # Move everything to the device
            features = features.to(device)
            targets = targets.to(device)

            # The decoder returns the reconstructions, latent means, and variances
            if model_type == "VariationalAutoEncoder":
                reconstructions, means, ln_variances = model(features)
            elif model_type == "AutoEncoder":
                reconstructions = model(features)
            reconstructions = reconstructions.to(device)
            loss = loss_function(reconstructions, targets)

            training_losses.append(loss.item())

    return {
        "error_avg": np.average(training_losses),
        "error_std": np.std(training_losses),
        "error_min": np.min(training_losses),
        "error_max": np.max(training_losses),
        "errors": training_losses
    }

