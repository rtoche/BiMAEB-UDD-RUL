import os
import json
import math
import pandas as pd
from torch.utils.data import Dataset


class BaseNormalityDataset(Dataset):
    def __init__(self,
                 data_dir_path: str,
                 dataset_csv_name: str,
                 window_length: int,
                 norm_op_len: int,
                 unit_identifier_col_name,
                 feature_names:list,
                 norm_op_as_pct: bool = False,
                 normalize: bool = False,
                 save_data: bool = False,
                 DEBUGGING: bool = False):

        if DEBUGGING:
            print("\nNASADataset object in DEBUGGING mode.")

        if window_length <= 0:
            error = "window_size must be greater than 0"
            raise RuntimeError(error)

        self.FEATURE_NAMES = feature_names
        self.number_ts_features = len(self.FEATURE_NAMES)
        self.__DEBUGGING__ = DEBUGGING
        self.normalize = normalize
        self.window_length = window_length
        self.norm_op_len = norm_op_len
        self.norm_op_as_pct = norm_op_as_pct
        self.path_to_dataset = os.path.join(data_dir_path, dataset_csv_name)

        # In order to make use of the .csv files, we must create the data based on the window_length variable.
        # The data will be stored as a JSON file. Check if this JSON file exists, if not, create
        # and save the file. This can be a computationally expensive process, thus the need to save it as a JSON file.
        # We only want the dataset name "FD00X", so take the first five chars [:5]
        dataset_name = f"normalized_{dataset_csv_name[:-4]}" if self.normalize else  dataset_csv_name[:-4]

        # Check if it's a percentage or set number of samples
        dataset_name = f"{dataset_name}_normOpPct{norm_op_len}" if self.norm_op_as_pct else f"{dataset_name}_normOpLen{norm_op_len}"

        dataset_name = f"{dataset_name}_window{self.window_length}"

        if self.__DEBUGGING__:
            JSON_file_name = f"{dataset_name}_DEBUGGING.json"
        else:
            JSON_file_name = f"{dataset_name}.json"

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

        if self.norm_op_as_pct:
            str_norm_op = f"{self.norm_op_len}%"
        else:
            str_norm_op = f"{self.norm_op_len}"

        print(f"{self.size} samples generated with norm_op_len={str_norm_op} and window_length={self.window_length}.")

    def __normalize_df__(self, df):
        df = df.copy()
        df_mins = df[self.FEATURE_NAMES].min()
        df_maxs = df[self.FEATURE_NAMES].max()

        df[self.FEATURE_NAMES] = (df[self.FEATURE_NAMES] - df_mins) / (df_maxs - df_mins + 1)

        return df

    def __create_JSON_from_window_size__(self,
                                         df_dataset: pd.DataFrame,
                                         path_to_json_file: str,
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
                unit_samples_list = self.__get_samples__(df_dataset,
                                                         unit=unit,
                                                         unit_identifier_col_name=unit_identifier_col_name)

                if len(unit_samples_list) > 0:
                    all_samples_list += unit_samples_list

                    if self.norm_op_as_pct:
                        print(f"Getting {self.norm_op_len}% of samples in unit {unit}: {len(unit_samples_list)}")
                    else:
                        print(f"Getting samples from unit {unit}: {len(unit_samples_list)}")
                    if self.__DEBUGGING__ and i == 0: break

        else:
            all_samples_list = self.__get_samples__(df_dataset,
                                                    unit=None,
                                                    unit_identifier_col_name=None)

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

    def __get_samples__(self,
                        df_dataset: pd.DataFrame,
                        unit_identifier_col_name: str = None,
                        unit: str = None):

        # Get the subset DataFrame for a unique unit
        if unit_identifier_col_name and unit:
            if type(unit) == int:
                query_string = f"{unit_identifier_col_name} == {unit}"
            elif type(unit) == str:
                query_string = f"{unit_identifier_col_name} == '{unit}'"
            df_dataset = df_dataset.query(query_string)

        # Adjust the total number of samples by making it a percentage of the dataset or a set number.
        if self.norm_op_as_pct:
            tot_num_samples = math.ceil(len(df_dataset) * (self.norm_op_len / 100))
        else:
            tot_num_samples = self.norm_op_len

        # Define maximum cycle
        max_start_cycle = df_dataset.shape[0] - self.window_length

        # If there are not enough samples, return an empty list.
        # The number of samples is equal to max_start_cycle
        if max_start_cycle < self.norm_op_len:
            return []

        # Perform basic error checking
        if self.window_length > tot_num_samples:
            error = "Window (window_size={}) must  be less than the number of total samples " \
                    "(unit_number_total_samples={}) for unit={}" \
                .format(self.window_length, tot_num_samples, unit)
            raise RuntimeError(error)

        # Compute the number of samples of size window_size that will be obtained form this unit.
        # Since we are using the sliding window approach, this can easily be computed.
        number_sliding_window_samples = tot_num_samples - self.window_length + 1

        # We start from the first cycle, which is at index=0
        list_all_samples = []
        for start_cycle_index in range(0, number_sliding_window_samples):
            # Make sure we're not going out of bounds with when doing the indexing.
            if start_cycle_index < max_start_cycle:
                # Only create window-slices of length window_length up until the normality_length
                # if start_cycle_index < self.normality_length:

                # Get the window as a dataframe (This may not be optimal since we are indexing a DataFrame)
                # Consider changing this to a 2D array?
                df_window = df_dataset.iloc[start_cycle_index:start_cycle_index + self.window_length, :]
                window = df_window.values.tolist()

                # Save the window as a list
                list_all_samples.append(window)

        return list_all_samples

    def __len__(self):
        return self.size
