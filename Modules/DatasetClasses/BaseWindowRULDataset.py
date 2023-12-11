import os
import json
import pandas as pd
from torch.utils.data import Dataset


class BaseWindowRULDataset(Dataset):
    def __init__(self,
                 data_dir_path: str,
                 dataset_csv_name: str,
                 time_column: str,
                 features: list,
                 window_length: int,
                 unit_identifier_col_name,
                 anomalous_col_identifier: str,
                 anomalous_identifier: str,
                 normalize: bool = True,
                 anomalies_only: bool = True,
                 save_data: bool = False,
                 DEBUGGING: bool = False):

        if DEBUGGING:
            print("\nDataset object in DEBUGGING mode.")

        if window_length <= 0:
            error = "window_size must be greater than 0"
            raise RuntimeError(error)

        self.DEBUGGING = DEBUGGING
        self.time_column = time_column
        self.features = features
        self.normalize = normalize
        self.window_length = window_length
        self.path_to_dataset = os.path.join(data_dir_path, dataset_csv_name)

        dataset_name = "normalized_{}".format(dataset_csv_name[:-4]) if self.normalize else dataset_csv_name[:-4]

        if anomalies_only:
            dataset_name = f"{dataset_name}_anomalies_only"
        else:
            dataset_name = f"{dataset_name}_all_samples"

        if self.DEBUGGING:
            JSON_file_name = "{}_{}_window{}.json".format("DEBUGGING", dataset_name, self.window_length)
        else:
            JSON_file_name = "{}_window{}.json".format(dataset_name, self.window_length)

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

            dict_all_samples_of_window_length = self.__create_JSON_from_window_size__(
                df_dataset=df_dataset,
                path_to_json_file=JSON_file_path,
                anomalous_col_identifier=anomalous_col_identifier,
                anomalous_identifier=anomalous_identifier,
                anomalies_only=anomalies_only,
                save_data=save_data,
                unit_identifier_col_name=unit_identifier_col_name
            )
        # Else, LOAD the data
        else:
            # Read file
            print("Reading {} JSON file at: \n{}\n".format(JSON_file_name, all_JSONs_files_dir_path))
            with open(JSON_file_path, 'r') as file:
                dict_all_samples_of_window_length = json.load(file)

        self.data = dict_all_samples_of_window_length["samples"]
        self.size = len(self.data)

        print("{} total samples generated with window_length={}."\
              .format(self.size, self.window_length))

    def __create_JSON_from_window_size__(self,
                                         df_dataset: pd.DataFrame,
                                         path_to_json_file: str,
                                         anomalous_col_identifier: str,
                                         anomalous_identifier: str,
                                         anomalies_only: bool,
                                         save_data: bool,
                                         unit_identifier_col_name: str = None):
        # If the dataset contains different units, each with their own cycles, iterate over each one
        # and collect samples for them
        if unit_identifier_col_name:
            # Iterate over all units and get their samples of window_length and save to list. Save them to a list
            all_samples_list: list = []

            # Get a list that contains the unique units in the list
            all_unique_units_list: list = df_dataset[unit_identifier_col_name].unique().tolist()

            for i, unit in enumerate(all_unique_units_list):
                df_unit = df_dataset.query("{} == {}".format(unit_identifier_col_name, unit))

                if anomalies_only:
                    df_unit_anomalies = df_unit.query(f"{anomalous_col_identifier}=={anomalous_identifier}")
                    anomaly_start_time = df_unit_anomalies[self.time_column].to_list()[0]
                    start_time = anomaly_start_time - self.window_length
                    df_unit = df_unit.iloc[start_time:, :]

                # Perform basic error checking
                if self.window_length > len(df_unit):
                    error = "Window (window_size={}) must  be less than the number of total samples " \
                            "(unit_number_total_samples={}) for unit={}" \
                        .format(self.window_length, len(df_unit), unit)
                    raise RuntimeError(error)

                unit_samples_list = self.__get_window_samples__(df_unit)
                all_samples_list += unit_samples_list

                print("Number of samples in unit {}: {}".format(unit, len(unit_samples_list)))
                if self.DEBUGGING and i == 0: break

        else:
            if anomalies_only:
                df_dataset = df_dataset.query(f"{anomalous_col_identifier}=={anomalous_identifier}")
            all_samples_list = self.__get_window_samples__(df_dataset)

        # Save the data
        dict_all_samples_of_window_length: dict = {
            "samples": all_samples_list,
        }

        if save_data:
            json_object = json.dumps(dict_all_samples_of_window_length)
            with open(path_to_json_file, 'w') as file:
                file.write(json_object)

            print("\nFinished writing JSON file to: \n{}.\n".format(path_to_json_file))

        return dict_all_samples_of_window_length

    def __get_window_samples__(self, df_dataset: pd.DataFrame):
        tot_num_samples = len(df_dataset)

        # Define maximum start index
        max_start_index = tot_num_samples - self.window_length + 1

        # Define total number of windows to extract
        number_window_samples = tot_num_samples - self.window_length + 1

        # We start from the first cycle, which is at index=0
        list_all_samples = []
        for start_cycle_index in range(0, number_window_samples):
            # Make sure we're not going out of bounds with when doing the indexing.
            if start_cycle_index < max_start_index:
                # Get the window as a dataframe (This may not be optimal since we are indexing a DataFrame)
                # Consider changing this to a 2D array?
                window = (df_dataset.iloc[start_cycle_index:start_cycle_index + self.window_length, :]).values.tolist()
                # window = df_window.values.tolist()

                # Save the window as a list
                list_all_samples.append(window)
        return list_all_samples

    def __normalize_df__(self, df):
        df = df.copy()
        df_mins = df[self.features].min()
        df_maxs = df[self.features].max()

        df[self.features] = (df[self.features] - df_mins) / (df_maxs - df_mins + 1)

        return df

    def __len__(self):
        return self.size
