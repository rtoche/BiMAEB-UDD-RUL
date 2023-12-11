import sys
import math
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

# Import Folder that has 'functions.py'
sys.path.insert(0, "/Users/rafaeltoche/Documents/School/Research/"\
                "Rainwaters_Lab/DART-LP2/Condition_Monitoring/DART-LP2/jupyter_notebooks/experiments/")
from functions import get_model_reconstructions


# ================================================================================================================
# Label a dataframe using the module2 that learned the normal behavior based on the "norm_op_len" which can either
# be a constant number of a percent of each time series.
# ================================================================================================================
def create_unsupervised_labeled_dataset(model, 
                                        df: pd.DataFrame, 
                                        identifier_col: str, 
                                        cycle_col: str, 
                                        fault_col: str,
                                        features_list: list,
                                        normal_op_len: int,
                                        normal_op_len_as_pct: bool,
                                        num_contiguous_anomalous_samples: int,
                                        num_stds: int, 
                                        unit_thresholds_dict: dict=None):
    
    # Get Reconstructions for each individual unit
    unique_units = df[identifier_col].unique()
    unsupervised_labeld_dfs = []
    
    if unit_thresholds_dict is None:
        thresholds_dict = {}
       
    
    num_units_with_anomalies = 0
    for i, unit in enumerate(unique_units):
        if type(unit) == str:
            query_str = f"{identifier_col}=='{unit}'"
        else:
            query_str = f"{identifier_col}=={unit}"
        df_unit = df.query(query_str)
        targets = df_unit[features_list].to_numpy()
        reconstructions = get_model_reconstructions(model, targets)

        # Get the number of normal samples for this unit
        if normal_op_len_as_pct:
            number_normal_samples = math.ceil((normal_op_len / 100) * len(df_unit))
        else:
            number_normal_samples = normal_op_len
        
        # Compute MSE, Average MSE up until 'number_normal_samples', and the STD of MSE
        mean_squared_recon_errors = np.square(reconstructions - targets).mean(axis=1)
        if number_normal_samples > 0:
            mse_mean = np.mean(mean_squared_recon_errors[:number_normal_samples])
            mse_std = np.std(mean_squared_recon_errors[:number_normal_samples])
        else:
            mse_mean = np.mean(mean_squared_recon_errors)
            mse_std = np.std(mean_squared_recon_errors)

        # Define the threshold
        if unit_thresholds_dict is None:
            # threshold = model_error_mean + 3 * model_error_std
            threshold = mse_mean + (num_stds * mse_std)
            thresholds_dict[str(unit)] = threshold
        else:
            threshold = unit_thresholds_dict[str(unit)]

        df_unit = label_df(df=df_unit,
                           reconstruction_errors=mean_squared_recon_errors, 
                           number_normal_samples=number_normal_samples, 
                           threshold=threshold, 
                           num_contiguous_anomalous_samples=num_contiguous_anomalous_samples)

        num_anomalies = len(df_unit.query(f"{fault_col}==1"))
        if num_anomalies > 0:
            num_units_with_anomalies += 1

        if number_normal_samples > 0:
            num_labeled_normal_samples = len(df_unit.query(f"{fault_col} == 0")) - number_normal_samples
            num_labeled_abnormal_samples = len(df_unit.query(f"{fault_col} == 1"))
        else:
            num_labeled_normal_samples = len(df_unit.query(f"{fault_col}== 0"))
            num_labeled_abnormal_samples = len(df_unit.query(f"{fault_col}== 1"))

        print(f"Unit {unit}")
        print(f"\t{'Training Samples:': <20} {number_normal_samples}")
        print(f"\t{'Normal Samples:': <20} {num_labeled_normal_samples}")
        print(f"\t{'Abnormal Samples:': <20} {num_labeled_abnormal_samples}\n")

        unsupervised_labeld_dfs.append(df_unit)
    
    print()
    print(f"Found {num_units_with_anomalies}/{len(unique_units)} units with anomalies")
        
    # df will contain two new colums, 'recon_errors' and 'fault'
    if unit_thresholds_dict is None:
        return pd.concat(unsupervised_labeld_dfs), thresholds_dict
    return pd.concat(unsupervised_labeld_dfs)

# ================================================================================================================
# Plot the latent space DataFrame, the Dataframe should be reduced down to 2 dimensions. Aditionally, it should 
# have the column to identify a system, the cycle, and whether it is normal or abnormal.
# ================================================================================================================
def plot_latent_space(df_latent: pd.DataFrame, title: str, x_label: str, y_label: str):
    dpi = 200
    width = 10
    height = 10
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    
    col1 = df_latent.columns.tolist()[3]
    col2 = df_latent.columns.tolist()[4]
    sns.scatterplot(x=col1, y=col2, data=df_latent, hue="label", palette=["#1f77b4", "red"], ax=ax)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    plt.show()

# ================================================================================================================
# Reconstructions Errors: The errors for the reconstructed sequence
# Normality length: Number of samples used to train the module2 (Num. samples assumed to operate normally)
# Threshold: The error threshold
# Num. Contiguous Anomalous Samples: Minimum number of samples in a row that must be above the threshold to
#         classify as the START of a fault/anomaly in the system.
# NOTE: It is assumed the 'number_normal_samples' is greater than num_anomaous_samples
# ================================================================================================================
def label_df(df: pd.DataFrame, 
             reconstruction_errors, 
             number_normal_samples: int, 
             threshold: float, 
             num_contiguous_anomalous_samples: int):
    assert len(reconstruction_errors) > number_normal_samples + num_contiguous_anomalous_samples, ("The length of 'reconstructions_errors' " + 
        f"must be greater than 'number_normal_samples + num_contiguous_anomalous_samples'")
    
    # Start iterating after 'number_normal_samples', since it is assumed that the first 'number_normal_samples'
    # are non-anomalous, by definition.
    fault_start = len(df)
    for i in range(number_normal_samples + num_contiguous_anomalous_samples, len(reconstruction_errors)):
        start = i-num_contiguous_anomalous_samples
        stop = i
        samples_window = reconstruction_errors[start:stop]
        
        # Check that no more than 'num_contiguous_anomalous_samples' are above the threshold
        num_samples_above_threshold = sum(samples_window > threshold)
        
        # If more than 'num_contiguous_anomalous_samples' samples are above the threshold,
        # then define this as the 'start' as the beginning of fault/anomaly and 
        # the rest of the samples, until the end of the sequence, are also labeled
        # as anomalous
        if num_samples_above_threshold >= num_contiguous_anomalous_samples:
            fault_start = start
            fault = True
            break
            
        # if not fault:
            # print(f"Range [{start}-{stop}]: Normal")
            
    anomalous_labels = [1] * (len(df) - fault_start)
    normal_lables = [0] * (len(df) - len(anomalous_labels))
            
    labels = normal_lables + anomalous_labels
    # print(f"Range [{start}-{stop}]: Anomaly/Fault Start")
    
    df = df.copy()
    df["recon_errors"] = reconstruction_errors
    df["fault"] = labels
    return df #, fault_start

# ================================================================================================================
# Function to plot the reconstruction errors
# ================================================================================================================
def plot_labels_recon_errors_and_threshold(model,
                                           number_normal_samples: int, # The number of samples used for training
                                           df: pd.DataFrame,
                                           threshold: float,
                                           features: list,
                                           title: str,
                                           recon_errors_title: str,
                                           threshold_title: str, 
                                           x_axis_column: str=None, 
                                           show_zoom_ins: bool=False):
    width = 10
    height = 8
    fig = plt.figure(figsize=(width,  height), dpi=200)
    ax = plt.subplot()
    
    targets = df[features].to_numpy()
    reconstructions = get_model_reconstructions(model, targets)
    
    # Compute MSE, Average MSE up until 'number_normal_samples', and the STD of MSE
    # mean_squared_recon_errors = np.square(reconstructions - targets).mean(axis=1)
    # mse_mean = np.mean(mean_squared_recon_errors[:number_normal_samples])
    # std = np.std(mean_squared_recon_errors[:number_normal_samples])
    # treshold = mse_mean + 3 * std
    
    if number_normal_samples > 0:
        df_normal_training = df.iloc[:number_normal_samples, :]
        df_normal = (df.iloc[number_normal_samples:, :]).query("fault==0")
    else:
        df_normal = (df.iloc[:, :]).query("fault==0")
    df_anomalous = df.query("fault==1")
    
    if number_normal_samples > 0:
        num_normal_training = len(df_normal_training)
    else: 
        num_normal_training = 0
        
    num_normal = len(df_normal)
    num_anomalous = len(df_anomalous)
    
    if num_anomalous > 0:
        degradation_start = len(df) - num_anomalous + 1
        # print(f"START: {degradation_start}")
    else:
        degradation_start = len(df)
    
    if show_zoom_ins:
        # Select range to zoom in on
        number_points = 15
        box_width = 0.25
        box_height = 0.2

        # Define the first Zoomed in Box
        x1_box1 = number_normal_samples - number_points
        x2_box1 = number_normal_samples + number_points
        y1_box = threshold - (threshold + threshold * 2)
        y2_box = threshold + (threshold + threshold * 2)
        axins_box1 = ax.inset_axes(
            [0.05, 0.3, box_width, box_height], # bounds[x0, y0, width, height]
            xlim=(x1_box1, x2_box1), ylim=(y1_box, y2_box), yticklabels=[])
            # xlim=(x1_box1, x2_box1), ylim=(y1_box, y2_box), xticklabels=[], yticklabels=[])

        # Define the second Zoomed in Box
        x1_box2 = degradation_start - number_points
        x2_box2 = degradation_start + number_points
        # y1_box = threshold - (threshold * 2)
        # y2_box2 = threshold + (threshold * 2)
        axins_box2 = ax.inset_axes(
            [0.4, 0.4, box_width, box_height], # bounds[x0, y0, width, height]
            # xlim=(x1_box2, x2_box2), ylim=(y1_box, y2_box2), xticklabels=[], yticklabels=[])
            xlim=(x1_box2, x2_box2), ylim=(y1_box, y2_box), yticklabels=[])
    
    if not x_axis_column:
        if num_normal_training > 0:
            x_axis = list(range(number_normal_samples))
           
            sns.lineplot(x=x_axis,
                            y="recon_errors", 
                            label=f"Normal Samples [n={number_normal_samples}]", 
                            color="green",  marker="o",
                            data=df_normal_training, 
                            ax=ax)
            
            if show_zoom_ins:
                sns.lineplot(x=x_axis,
                             y="recon_errors", 
                             color="green", 
                             marker="o", 
                             data=df_normal_training,
                             ax=axins_box1)

                sns.lineplot(x=x_axis,
                             y="recon_errors", 
                             color="green", 
                             marker="o", 
                             data=df_normal_training,
                             ax=axins_box2)
            
        if num_normal > 0:
            x_axis = list(range(number_normal_samples, number_normal_samples + num_normal))
                        
            sns.lineplot(x=x_axis, y="recon_errors", 
                         label="Unsupervised Labels: Normal", marker="o", data=df_normal, ax=ax)
            
            if show_zoom_ins:
                sns.lineplot(x=x_axis,
                             y="recon_errors",  
                             marker="o", 
                             data=df_normal,
                             ax=axins_box1)

                sns.lineplot(x=x_axis,
                             y="recon_errors",  
                             marker="o", 
                             data=df_normal,
                             ax=axins_box2)
            
        if num_anomalous > 0:
            x_axis = list(range(number_normal_samples + num_normal, num_normal_training + num_normal + num_anomalous))
            
            sns.lineplot(x=x_axis, 
                            y="recon_errors", 
                            label=f"Unsupervised Labels: Abnormal [Degradation Start={degradation_start}]", 
                            color="red",  marker="o",
                            data=df_anomalous, 
                            ax=ax)
            
            if show_zoom_ins:
                sns.lineplot(x=x_axis,
                             y="recon_errors",  
                             color="red",
                             marker="o", 
                             data=df_anomalous,
                             ax=axins_box1)

                sns.lineplot(x=x_axis,
                             y="recon_errors",  
                             color="red",
                             marker="o", 
                             data=df_anomalous,
                             ax=axins_box2)
        
        x_axis = list(range(len(df)))
        sns.lineplot(x=x_axis, y=threshold, label="Threshold", color="orange", data=df, ax=ax)
        
    else:
        if num_normal_training > 0:
            sns.lineplot(x=x_axis_column, 
                            y="recon_errors", label=f"Training Samples [n={number_normal_samples}]", 
                            color="green", marker="o", data=df_normal_training, ax=ax)
            
            if show_zoom_ins:
                sns.lineplot(x=x_axis_column,
                             y="recon_errors",  
                             color="green",
                             marker="o", 
                             data=df_normal_training,
                             ax=axins_box1)
                sns.lineplot(x=x_axis_column,
                             y="recon_errors",  
                             color="green",
                             marker="o", 
                             data=df_normal_training,
                             ax=axins_box2)

        if num_normal > 0:            
            sns.lineplot(x=x_axis_column, y="recon_errors", 
                         label="Unsupervised Labels: Normal", marker="o", data=df_normal, ax=ax)
            
            if show_zoom_ins:
                sns.lineplot(x=x_axis_column,
                             y="recon_errors",  
                             marker="o", 
                             data=df_normal,
                             ax=axins_box1)

                sns.lineplot(x=x_axis_column,
                             y="recon_errors",  
                             marker="o", 
                             data=df_normal,
                             ax=axins_box2)
        
        if num_anomalous > 0:            
            sns.lineplot(x=x_axis_column, y="recon_errors", 
                         label=f"Unsupervised Labels: Abnormal [Degradation Start={degradation_start}]", 
                         color="red",  marker="o", data=df_anomalous, ax=ax)
            
            if show_zoom_ins:
                sns.lineplot(x=x_axis_column,
                             y="recon_errors",  
                             color="red",
                             marker="o", 
                             data=df_anomalous,
                             ax=axins_box1)

                sns.lineplot(x=x_axis_column,
                             y="recon_errors",  
                             color="red",
                             marker="o", 
                             data=df_anomalous,
                             ax=axins_box2)
        
        sns.lineplot(x=x_axis_column, y=threshold, label=threshold_title, color="orange", data=df, ax=ax)
        if show_zoom_ins:
            sns.lineplot(x=x_axis_column, y=threshold, color="orange", data=df,ax=axins_box1)
            sns.lineplot(x=x_axis_column, y=threshold, color="orange", data=df,ax=axins_box2)
    
    ax.set_title(title)
    ax.set_ylabel(recon_errors_title)
    if not x_axis_column:
        ax.set_xlabel("Cycle")
    else:
        ax.set_xlabel(x_axis_column)
    
    if show_zoom_ins:
        # Get rid of ticks and labels on both boxes
        axins_box1.set_xlabel("")
        axins_box1.set_ylabel("")
        axins_box1.patch.set_edgecolor("black")  
        axins_box1.patch.set_linewidth(2)  

        axins_box2.set_xlabel("")
        axins_box2.set_ylabel("")
        axins_box2.patch.set_edgecolor("black")  
        axins_box2.patch.set_linewidth(2)  

        # Define Inset Zoom for boxes
        ax.indicate_inset_zoom(axins_box1, edgecolor="black")   
        ax.indicate_inset_zoom(axins_box2, edgecolor="black")   
        
    plt.show()





