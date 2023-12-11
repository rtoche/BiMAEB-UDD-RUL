import os
import json
import torch
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

sns.set_theme()

TURBOFAN_FEATURES_LIST = ["operational_setting_1","operational_setting_2",
                          "operational_setting_3","sensor_measurement_1",
                          "sensor_measurement_2","sensor_measurement_3",
                          "sensor_measurement_4","sensor_measurement_5",
                          "sensor_measurement_6","sensor_measurement_7",
                          "sensor_measurement_8","sensor_measurement_9",
                          "sensor_measurement_10","sensor_measurement_11",
                          "sensor_measurement_12","sensor_measurement_13",
                          "sensor_measurement_14","sensor_measurement_15",
                          "sensor_measurement_16","sensor_measurement_17",
                          "sensor_measurement_18","sensor_measurement_19",
                          "sensor_measurement_20","sensor_measurement_21"]

NASA_TURBOFAN_DATASET = "NASA_Turbofan_dataset"
AE_MODEL_TYPE = "AutoEncoder"
VAE_MODEL_TYPE = "VariationalAutoEncoder"


def view_performance_on_data(model, df: pd.DataFrame, 
                             identifier_col: str, 
                             cycle_col: str, 
                             rul_col: str, 
                             fault_col: str, 
                             show_anomalies_only: bool = False, 
                             show_scores: bool = False):
    
    # Get Reconstructions for each individual unit
    unique_units = df[identifier_col].unique()

    all_mses = []
    mses_for_data_with_fault_label = []
    mses_for_data_without_fault_label = []

    all_scores = 0
    score_data_with_fault_label = 0
    score_data_without_fault_label = 0

    num_units_with_anomalies = 0

    for i, unit in enumerate(unique_units): 
        df_unit = df.query(f"{identifier_col}=={unit}")
        df_unit_last_cycle = df_unit.iloc[-1:, :]

        cycle = df_unit_last_cycle[cycle_col].to_numpy()[0]
        features = df_unit_last_cycle[TURBOFAN_FEATURES_LIST].to_numpy()
        targets = df_unit_last_cycle[rul_col].to_numpy()
        predictions = get_model_rul_predictions(model, features)

        # Compute Average MSE
        mse = np.square(predictions - targets).mean()
        # Compute all_scores
        unit_score = get_scores(predictions, targets)
        all_scores += unit_score

        # Save MSE for all data
        all_mses.append(mse)

        num_unit_anomalies = len(df_unit.query(f"{fault_col}==1"))

        if (show_anomalies_only and num_unit_anomalies > 0):
            print(f"Computing RUL for Unit {unit} (on cycle={cycle})")
            for i in range(len(targets)):
                print(f"Target: {targets[i]} --> Pred.: {predictions[i]: .0f}")
                if show_scores:
                    print(f"MSE={mse:0.2f}")
                    print(f"Score={unit_score:0.2f}")
            print()
        elif(not show_anomalies_only):
            print(f"Computing RUL for Unit {unit} (on cycle={cycle})")
            for i in range(len(targets)):
                print(f"Target: {targets[i]} --> Pred.: {predictions[i]: .0f}")
                if show_scores:
                    print(f"MSE={mse:0.2f}")
                    print(f"Score={unit_score:0.2f}")
            print()

        if num_unit_anomalies > 0:
            num_units_with_anomalies += 1
            df_unit = df_unit.query(f"{fault_col}==1")
            mses_for_data_with_fault_label.append(mse)
            score_data_with_fault_label += unit_score
        else:
            # print(f"{cycle_col}={cycle}")
            mses_for_data_without_fault_label.append(mse)
            score_data_without_fault_label += unit_score

    # Compute Overall Averages
    all_data_average_mse = np.mean(all_mses)
    all_data_median_mse = np.median(all_mses)

    average_mse_fault_data = np.mean(mses_for_data_with_fault_label)
    median_mse_fault_data = np.median(mses_for_data_with_fault_label)

    average_mse_no_fault_data = np.mean(mses_for_data_without_fault_label)
    median_mse_no_fault_data = np.median(mses_for_data_without_fault_label)

    print("================================================")
    print(f"- RMSE on Validation Data ALL Faults: {np.sqrt(all_data_average_mse): .6f}")
    print(f"- Score on ALL Validation Data: {all_scores: .6f}")
    print(f"- Average MSE on ALL Validation Data: {all_data_average_mse: .6f}")
    print(f"- Median MSE on ALL Validation Data: {all_data_median_mse: .6f}\n")

    print(f"# - RMSE on Validation Data WITH Faults: {np.sqrt(average_mse_fault_data): .6f}")
    print(f"# - Score on Validation Data WITH Faults: {score_data_with_fault_label: .6f}")
    print(f"- Average MSE on Validation Data WITH Faults: {average_mse_fault_data: .6f}")
    print(f"- Median MSE on Validation Data WITH Faults: {median_mse_fault_data: .6f}\n")


    print(f"- Average MSE on Validation Data WITHOUT Faults: {average_mse_no_fault_data: .6f}")
    print(f"- Score on Validation Data WITHOUT Faults: {score_data_without_fault_label: .6f}")
    print(f"- Median MSE on Validation Data WITHOUT Faults: {median_mse_no_fault_data: .6f}\n")
    
    print(f"- Found {num_units_with_anomalies} units with anomalies out of {len(unique_units)}")

    
def view_RUL_stats(df):
    RUL_min = df["RUL"].min()
    RUL_max = df["RUL"].max()
    RUL_mean = df["RUL"].mean()
    RUL_median = df["RUL"].median()

    print(f"RUL Min: {RUL_min}")
    print(f"RUL Max: {RUL_max}")
    print(f"RUL Mean: {RUL_mean}")
    print(f"RUL Median: {RUL_median}")
    
    sns.displot(x="RUL", data=df)
    

def save_labeled_dataframe(file_name: str, 
                           save_path: str, 
                           df_labeled: pd.DataFrame, 
                           df_unlabeled: pd.DataFrame):
    df_unlabeled = df_unlabeled.copy()
    
    pth = os.path.join(save_path, file_name)
    
    df_unlabeled["fault"] = df_labeled["fault"]
    df_unlabeled.to_csv(pth, index=False)

    print(f"Saved '{file_name}' to -->")
    print(f"'{save_path}'")
    
    
def get_mse(predictions, targets):
    return np.square(predictions - targets).mean()

    
def get_scores(predictions, targets):
    d = predictions - targets
    
    scores_to_sum = []
    for d_i in d:
        if d_i < 0:
            exp_score = np.exp(-d_i/10) - 1
        elif d_i >= 0:
            exp_score = np.exp(d_i/13) - 1
        scores_to_sum.append(exp_score)
    scores_to_sum = np.asarray(scores_to_sum)
    return scores_to_sum.sum()


def get_model_lstm_rul_predictions(model, features):
    # Convert DataFrame to Numpy and then Torch Tensor so they can be fed into module2
    features_expanded_dims = np.expand_dims(features, axis=0).astype(np.float32)
    model_input_features = torch.from_numpy(features_expanded_dims)
    
    preds = model(model_input_features)
    
    return preds.detach().numpy()[:, 0]


def get_model_rul_predictions(model, features: np.asarray):
    # Convert DataFrame to Numpy and then Torch Tensor so they can be fed into module2
    # model_input_features = np.expand_dims(features, axis=1).astype(np.float32)
    model_input_features = torch.from_numpy(features.astype(np.float32))
    
    preds = model(model_input_features)
    
    return preds.detach().numpy()[:, 0]


def read_errors_stats_dict(file_name):
    with open(file_name)  as file:
        errors_stats_dict = json.load(file)
    
    return errors_stats_dict
    

def normalize_df_with_context(df, df_context, model_features: list):
    df = df.copy()
    df_mins = df_context[model_features].min()
    df_maxs = df_context[model_features].max()

    df[model_features] = (df[model_features] - df_mins) / (df_maxs - df_mins + 1)

    return df
    
def normalize_df(df, model_features: list):
    df = df.copy()
    # df_mins = df.iloc[:, 2:-1].min()
    # df_maxs = df.iloc[:, 2: -1].max()
    df_mins = df[model_features].min()
    df_maxs = df[model_features].max()

    # df_cols = df.iloc[model_features].columns
    # df[model_features] = (df.iloc[:, 2:-1] - df_mins) / (df_maxs - df_mins + 1)
    df[model_features] = (df[model_features] - df_mins) / (df_maxs - df_mins + 1)

    return df


def get_model_reconstructions(model, targets: np.asarray):
    # Convert DataFrame to Numpy and then Torch Tensor so they can be fed into module2
    model_input_targets = np.expand_dims(targets, axis=1).astype(np.float32)
    model_input_targets = torch.from_numpy(model_input_targets)
    
    if model.model_type == AE_MODEL_TYPE:
        reconstructions = model(model_input_targets)
    elif model.model_type == VAE_MODEL_TYPE:
        reconstructions, _, _ = model(model_input_targets)
        
    return reconstructions.detach().numpy()[:, 0, :]


def plot_feature_recon(model, 
                       window_recon_size: int, 
                       df: pd.DataFrame, 
                       col_name: str, 
                       model_features: list,
                       unit: int=None):
    width = 10
    height = 8
    fig = plt.figure(figsize=(width, height))
    ax = plt.subplot()

    # Get unit and get all but the first 2 cols and the last one
    if unit:
        df = df.query("unit=={}".format(unit))

    # Get only the window we want to reconstruct, and exclude all columns that are not
    # used for prediction
    df = df[model_features]
    df = df.iloc[0:window_recon_size, :]

    window_length = df.shape[0]

    # range(start, stop, step)
    reconstructed_feature = []
    for i in range(0, window_length): 
        target_sample = np.asarray([df.iloc[i, :].to_numpy()])

        # Convert DataFrame to Numpy and then Torch Tensor so they can be fed into module2
        model_input_features = np.expand_dims(target_sample, axis=0).astype(np.float32)
        model_input_features = torch.from_numpy(model_input_features)

        # Get module2 reconstructions
        model.eval()
        with torch.no_grad():
            if model.model_type == AE_MODEL_TYPE:
                reconstructed_sample = model(model_input_features)
            elif model.model_type == VAE_MODEL_TYPE:
                reconstructed_sample, _, _ = model(model_input_features)
        reconstructed_sample = reconstructed_sample.detach().numpy()[0]

        # Get the reconstructed feature
        # col_index = __turbofan_col_name_to_index(col_name)
        col_index = df.columns.get_loc(col_name)

        recon_feature = reconstructed_sample[:, col_index][0]

        # Save the reconstructed feature for this time step
        reconstructed_feature.append(recon_feature)

    target_feature = df[col_name].to_numpy()

    # Plot the reconstructed feature/true features
    x_axis = list(range(0, window_length))
    sns.lineplot(x=x_axis, y=target_feature, label="Target Feature", ax=ax)
    sns.lineplot(x=x_axis, y=reconstructed_feature, label="Reconsturcted Feature", ax=ax)
    ax.set_title(col_name)

    # print("Metrics for {}".format(col_name))
    # print("Getting windows from ({}, {})".format(0, window_length))
    # print("MSE: {}".format(np.square(reconstructed_feature - target_feature).mean()))
    # print("Target Feature Average Vaue: {}".format(np.mean(target_feature)))
    # print("Predicted Feature Average Vaue: {}\n".format(np.mean(reconstructed_feature)))

    
def plot_rul_predictions_and_targets(predictions, targets, cycles, cycle_length=50):
    # Compute Score
    score = get_scores(predictions, targets)
    mse = get_mse(predictions, targets)
    print(f"Score: {score}")
    print(f"MSE: {mse}")
    
    width = 10
    height = 8
    fig = plt.figure(figsize=(width,  height), dpi=200)
    ax = plt.subplot()
    x_axis = cycles
    
    sns.lineplot(x=x_axis, y=targets, label="True RUL", ax=ax)
    sns.lineplot(x=x_axis, y=predictions, label="Predicted RUL", ax=ax)
    
    ax.set_ylabel("RUL")
    ax.set_xlabel("Cycle")
    plt.show()
    

def plot_recon_errors_and_threshold(errors: list,
                                    error_threshold: float,
                                    recon_errors_title: str,
                                    threshold_title: str):
    width = 10
    height = 8
    fig = plt.figure(figsize=(width,  height), dpi=200)
    ax = plt.subplot()
    x_axis = list(range(0, len(errors)))
    
    sns.lineplot(x=x_axis, y=errors, label=recon_errors_title, ax=ax)
    sns.lineplot(x=x_axis, y=error_threshold, label=threshold_title, ax=ax)
    ax.set_ylabel("Reconstruction Error")
    ax.set_xlabel("Cycle")
    plt.show()


def plot_unit_recon_error_and_threshold(unit: int, 
                                        normality_length: int, 
                                        recons_dict: dict, 
                                        std: int = 3):
    mean_squared_recon_errors = recons_dict[unit]["mean_squared_reconstruction_errors"]
    quant_errors = recons_dict[unit]["quant_errors"]
    
    # Compute Averages/STDs on TRAINING samples
    avg_mse_on_training_samples = np.mean(mean_squared_recon_errors[:normality_length])
    training_mse_std = np.std(mean_squared_recon_errors[:normality_length])
    avg_quant_error_on_training_samples = np.mean(quant_errors[:normality_length])
    training_quant_err_std = np.std(quant_errors[:normality_length])
    
    # Compute Averages/STDs on UNSEEN samples
    avg_mse_on_unseen_samples = np.mean(mean_squared_recon_errors[normality_length:])
    avg_quant_error_on_unseen_samples = np.mean(quant_errors[normality_length:])
    
    # Defien the threshold from the TRAINING errors
    mse_threshold = avg_mse_on_training_samples + std * training_mse_std
    eucledian_threshold = avg_quant_error_on_training_samples + std * training_quant_err_std
    
    print(f"Unit Average MSE on TRAINING Data(cycles > {normality_length}): {avg_mse_on_training_samples}")
    print(f"Unit Average Quant. Error on TRAINING Data(cycles > {normality_length}): {avg_quant_error_on_training_samples}\n")
    
    print(f"Unit Average MSE on UNSEEN Data(cycles < {normality_length}): {avg_mse_on_unseen_samples}")
    print(f"Unit Average Quant. Error on UNSEEN Data(cycles < {normality_length}): {avg_quant_error_on_unseen_samples}\n")

    print(f"MSE Threshold: {mse_threshold}")
    print(f"Quant. Error Threshold: {eucledian_threshold}")
    
    plot_recon_errors_and_threshold(mean_squared_recon_errors,
                                    mse_threshold,
                                    recon_errors_title="Mean Squared Reconstruction Errors",
                                    threshold_title="MSE Threshold")
    
    plot_recon_errors_and_threshold(quant_errors,
                                    eucledian_threshold,
                                    recon_errors_title="Eucledian Reconstruction Errors",
                                    threshold_title="Eucledian Threshold")
    
    
def plot_labels_recon_errors_and_threshold_on_validation_data(model,
                                                              df: pd.DataFrame,
                                                              features: list,
                                                              threshold: float,
                                                              title: str,
                                                              recon_errors_title: str,
                                                              threshold_title: str, 
                                                              x_axis_column: str=None):
    width = 10
    height = 8
    fig = plt.figure(figsize=(width,  height), dpi=200)
    ax = plt.subplot()
    
    targets = df[features].to_numpy()
    reconstructions = get_model_reconstructions(model, targets)
    
    # Compute MSE, Average MSE up until 'normality_length', and the STD of MSE
    mean_squared_recon_errors = np.square(reconstructions - targets).mean(axis=1)
    mse_mean = np.mean(mean_squared_recon_errors)
    mse_std = np.std(mean_squared_recon_errors)
    
    df_normal = df.query("fault==0")
    df_anomalous = df.query("fault==1")

    num_normal = len(df_normal)
    num_anomalous = len(df_anomalous)
    
    if not x_axis_column:
        x_axis = list(range(num_normal))
        sns.scatterplot(x=x_axis, y="recon_errors", 
                        label="Unsupervised Labeled: Normal", data=df_normal, ax=ax)
                
        if num_anomalous:
            x_axis = list(range(num_normal, num_normal + num_anomalous))
            start = x_axis[0]
            sns.scatterplot(x=x_axis, y="recon_errors", 
                            label=f"Unsupervised Labeled: Anomalous [start={start}]", 
                            color="red", data=df_anomalous, ax=ax)
        
        x_axis = list(range(len(df)))
        sns.lineplot(x=x_axis, y=threshold, label="Threshold", color="orange", data=df, ax=ax)
        
    else:
        sns.scatterplot(x=x_axis_column, y="recon_errors", 
                        label="Unsupervised Labels: Normal", data=df_normal, ax=ax)
        
        if num_anomalous > 0:
            start = df_anomalous[x_axis_column].to_list()[0]
            sns.scatterplot(x=x_axis_column, y="recon_errors", 
                            label=f"Unsupervised Labels: Anomalous [start={start}]", 
                            color="red", data=df_anomalous, ax=ax)
            
        sns.lineplot(x=x_axis_column, y=threshold, label=threshold_title, color="orange", data=df, ax=ax)
    
        
    ax.set_title(title)
    ax.set_ylabel(recon_errors_title)
    if not x_axis_column:
        ax.set_xlabel("Cycle")
    else:
        ax.set_xlabel(x_axis_column)
    plt.show()
    

def get_latent_space_df(model, 
                        df: pd.DataFrame, 
                        unit_col_identifier_name: str, 
                        cycle_col: str,
                        features_list: list):
    latent_space_samples = []
    # labels = []
    # cycles = []
    unit_labels = []

    unit_samples = df[features_list].to_numpy()

    # Normal Samples Reshaping
    model_input_samples = np.expand_dims(unit_samples, axis=1).astype(np.float32)
    model_input_samples = torch.from_numpy(model_input_samples)

    # Get the latent space representation for normal samples
    if model.model_type == "VariationalAutoEncoder":
        z, _, _ = model.latent_space(model.encoder(model_input_samples))
    elif model.model_type == "AutoEncoder":
        z = model.latent_space(model.encoder(model_input_samples))
    else:
        print("Model must be a VariationalAutoEncoder or AutoEncoder")
        return

    latent_space_samples = torch.squeeze(z, dim=1).detach().numpy()
    
    labels = ["normal" if i == 0 else "abnormal" for i in df["fault"].to_list()]
    unit_labels = df[f"{unit_col_identifier_name}"].to_list()
    cycles = df[f"{cycle_col}"].to_list()

    if latent_space_samples.shape[1] > 2:
        col1 = "x1"
        col2 = "x2"
        tsne = TSNE(n_components=2)
        latent_space_samples_in_2D = tsne.fit_transform(latent_space_samples)
        data = latent_space_samples_in_2D
    else:
        col1 = "z[0]"
        col2 = "z[1]"
        data = latent_space_samples

    df_latent_space_2D = pd.DataFrame(data, columns=[col1, col2])    
    df_latent_space_2D.insert(0, f"{unit_col_identifier_name}", unit_labels)
    df_latent_space_2D.insert(1, f"{cycle_col}", cycles)
    df_latent_space_2D.insert(2, "label", labels)
    
    return df_latent_space_2D