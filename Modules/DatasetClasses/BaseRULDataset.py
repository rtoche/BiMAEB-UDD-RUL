import os
import json

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class BaseRULDataset(Dataset):
    def __init__(self,
                 data_dir_path: str,
                 dataset_csv_name: str,
                 unit_identifier_col_name,
                 anomalous_col_identifier: str,
                 anomalous_identifier: str,
                 features_list: list,
                 normalize: bool = False,
                 anomalies_only: bool = True,
                 save_data: bool = False,
                 DEBUGGING: bool = False):

        if DEBUGGING:
            print("\nNASADataset object in DEBUGGING mode.")

        self.__DEBUGGING__ = DEBUGGING
        self.normalize = normalize
        self.path_to_dataset = os.path.join(data_dir_path, dataset_csv_name)
        self.anomalies_only = anomalies_only
        self.FEATURES_LIST = features_list

        # In order to make use of the .csv files, we must create the data based on the window_length variable.
        # The data will be stored as a JSON file. Check if this JSON file exists, if not, create
        # and save the file. This can be a computationally expensive process, thus the need to save it as a JSON file.
        # We only want the dataset name "FD00X", so take the first five chars [:5]
        csv_file_name = dataset_csv_name[:-4]

        dataset_name = "{}_normalized".format(csv_file_name) if self.normalize else csv_file_name

        dataset_name = f"{dataset_name}_anomalies_only" if self.anomalies_only else f"{dataset_name}_all_samples"

        if self.__DEBUGGING__:
            JSON_file_name = "{}_{}_RUL.json".format("DEBUGGING", dataset_name)
        else:
            JSON_file_name = "{}_RUL.json".format(dataset_name)

        # Define the directory where all JSON files will be stored.
        all_JSONs_files_dir_path = os.path.join(data_dir_path, "JSON_data")
        JSON_file_path = os.path.join(all_JSONs_files_dir_path, JSON_file_name)

        # If NOT exists, CREATE the data
        if not os.path.exists(JSON_file_path):
            if not os.path.exists(all_JSONs_files_dir_path):
                os.makedirs(all_JSONs_files_dir_path)
            # Read data files only if the JSON file does not exist. These are big files so no need to read
            # them if there is no need.
            print("JSON file '{}' does not exist at {}. \n\nCreating JSON file...".format(JSON_file_name,
                                                                                          JSON_file_path))
            df_dataset: pd.DataFrame = pd.read_csv(self.path_to_dataset)

            # Normalize all columns in the DataFrame between 0 and 1
            if self.normalize:
                df_dataset = self.__normalize_df__(df_dataset)

            dict_all_anomalous_samples = self.__get_anomalous_samples__(
                df_dataset=df_dataset,
                path_to_json_file=JSON_file_path,
                anomalous_col_identifier=anomalous_col_identifier,
                anomalous_identifier=anomalous_identifier,
                save_data=save_data,
                unit_identifier_col_name=unit_identifier_col_name
            )
        # Else, LOAD the data
        else:
            # Read file
            print("Reading {} JSON file at: \n{}\n".format(JSON_file_name, all_JSONs_files_dir_path))
            with open(JSON_file_path, 'r') as file:
                dict_all_anomalous_samples = json.load(file)

        self.data = dict_all_anomalous_samples["samples"]
        self.size = len(self.data)

        print(f"{self.size} total samples anomalous samples.")

    def __normalize_df__(self, df):
        df = df.copy()
        df_mins = df[self.FEATURES_LIST].min()
        df_maxs = df[self.FEATURES_LIST].max()

        df[self.FEATURES_LIST] = (df[self.FEATURES_LIST] - df_mins) / (df_maxs - df_mins + 1)

        return df

    def __get_anomalous_samples__(self,
                                  df_dataset: pd.DataFrame,
                                  path_to_json_file: str,
                                  anomalous_col_identifier: str,
                                  anomalous_identifier: str,
                                  save_data: bool,
                                  unit_identifier_col_name: str = None):
        all_samples_list: list = []
        # If the dataset contains different units, each with their own cycles, iterate over each one
        # and collect samples for them
        if unit_identifier_col_name:
            all_unique_units_list: list = df_dataset[unit_identifier_col_name].unique().tolist()
            for i, unit in enumerate(all_unique_units_list):
                if type(unit) == int:
                    query_string = f"{unit_identifier_col_name}=={unit}"
                else:
                    query_string = f"{unit_identifier_col_name}=='{unit}'"
                df_unit: pd.DataFrame = (df_dataset.query(query_string))

                if self.anomalies_only:
                    unit_samples_list = (df_unit.query(f"{anomalous_col_identifier}=={anomalous_identifier}")).to_numpy()
                else:
                    unit_samples_list = df_unit.to_numpy()

                all_samples_list += unit_samples_list.tolist()

                print("Number of samples in unit {}: {}".format(unit, len(unit_samples_list)))
                if self.__DEBUGGING__ and i == 0: break

        else:
            if self.anomalies_only:
                samples = (df_dataset.query(f"{anomalous_col_identifier}=={anomalous_identifier}")).to_numpy()
            else:
                samples = df_dataset.to_numpy()

            all_samples_list += samples.tolist()

        # Save the data
        dict_all_anomalous_samples: dict = {
            "samples": all_samples_list,
        }

        if save_data:
            json_object = json.dumps(dict_all_anomalous_samples)
            with open(path_to_json_file, 'w') as file:
                file.write(json_object)

            print("\nFinished writing JSON file to: \n{}.\n".format(path_to_json_file))

        return dict_all_anomalous_samples

    def __len__(self):
        return self.size
