import os
import numpy as np
from Modules.DatasetClasses.BaseRULDataset import BaseRULDataset


class RULTurbofanDataset(BaseRULDataset):
    def __init__(self,
                 data_dir_path: str,
                 dataset_csv_name: str,
                 normalize: bool = False,
                 anomalies_only: bool = True,
                 DEBUGGING: bool = False):
        # Define the column in the DataFrame that identifies the column with fault
        # and the value in the column that designates a fault.
        self.FEATURES_LIST = [
            "cycle", "operational_setting_1", "operational_setting_2", "operational_setting_3",
            "sensor_measurement_1", "sensor_measurement_2", "sensor_measurement_3", "sensor_measurement_4",
            "sensor_measurement_5", "sensor_measurement_6", "sensor_measurement_7", "sensor_measurement_8",
            "sensor_measurement_9", "sensor_measurement_10", "sensor_measurement_11", "sensor_measurement_12",
            "sensor_measurement_13", "sensor_measurement_14", "sensor_measurement_15", "sensor_measurement_16",
            "sensor_measurement_17", "sensor_measurement_18", "sensor_measurement_19", "sensor_measurement_20",
            "sensor_measurement_21"]

        unit_identifier_col_name = "unit"
        anomalous_col_identifier = "fault"
        anomalous_identifier = "1"
        super(RULTurbofanDataset, self).__init__(data_dir_path=data_dir_path,
                                                 dataset_csv_name=dataset_csv_name,
                                                 unit_identifier_col_name=unit_identifier_col_name,
                                                 anomalous_col_identifier=anomalous_col_identifier,
                                                 anomalous_identifier=anomalous_identifier,
                                                 features_list=self.FEATURES_LIST,
                                                 normalize=normalize,
                                                 anomalies_only=anomalies_only,
                                                 DEBUGGING=DEBUGGING)

        self.number_ts_features = len(self.FEATURES_LIST)

    def __getitem__(self, idx):
        # Every sample in the data array should have the following structure:
        # [
        #   "unit","cycle","operational_setting_1","operational_setting_2","operational_setting_3",
        #   "sensor_measurement_1", "sensor_measurement_2","sensor_measurement_3","sensor_measurement_4",
        #   "sensor_measurement_5","sensor_measurement_6", "sensor_measurement_7","sensor_measurement_8",
        #   "sensor_measurement_9","sensor_measurement_10", "sensor_measurement_11", "sensor_measurement_12",
        #   "sensor_measurement_13","sensor_measurement_14", "sensor_measurement_15","sensor_measurement_16",
        #   "sensor_measurement_17","sensor_measurement_18", "sensor_measurement_19","sensor_measurement_20",
        #   "sensor_measurement_21","RUL", "<FAULT_COLUMN>"
        # ]
        # sample = np.array(self.data[idx], dtype=np.float32)
        sample = self.data[idx]

        # Create the features used for training and set as float32.
        # Also create the labels (RUL) which include system name for labeling purposes.
        # Features are all columns except the first two (Unit and cycle) and last one (RUL),
        # RUL = last column
        # TODO: Perhaps also return the 'cycle' as a feature?
        unit = int(sample[0])
        # features = np.asarray(sample[2:-2], dtype=np.float32)
        features = np.asarray(sample[1:-2], dtype=np.float32)
        rul = np.asarray(sample[-2], dtype=np.float32)
        # fault = sample[-1]

        return [features, rul]


# =============================================================== #
# The code below is used to test the implemented dataset.         #
# class above.                                                    #
# =============================================================== #
def __test__():
    import math
    from torch.utils.data import DataLoader
    from torch.utils.data import random_split

    root_data_dir_path = "/Users/rafaeltoche/Documents/School/Research/" \
                         "Rainwaters_Lab/DART-LP2/Condition_Monitoring/NASA_turbofan_data/train"
    csv_dataset_name = "FD001_train_unsupervised_labels.csv"

    print("Reading data from: \n{}".format(os.path.join(root_data_dir_path, csv_dataset_name)))

    NASA_dataset = RULTurbofanDataset(data_dir_path=root_data_dir_path,
                                      dataset_csv_name=csv_dataset_name,
                                      normalize=True,
                                      anomalies_only=False,
                                      DEBUGGING=False)

    split = 0.80
    train_size = math.ceil(NASA_dataset.size * split)
    test_size = NASA_dataset.size - train_size

    print("Train Size: {}".format(train_size))
    print("Test Size: {}".format(test_size))

    # Define train and test split
    train_dataset, test_dataset = random_split(NASA_dataset, [train_size, test_size])

    # Define the DataLoaders
    batch_size = 5
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train_features, train_RUL_labels = next(iter(train_dataloader))

    print("Features Shape: {}".format(train_features.shape))
    print("Labels Shape: {}\n".format(train_RUL_labels.shape))

    print("Features Dtype: {}".format(train_features.dtype))
    print("Labels Dtype: {}".format(train_RUL_labels.dtype))


if __name__ == "__main__":
    __test__()

