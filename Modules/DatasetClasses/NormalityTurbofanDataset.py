import os
import numpy as np
from Modules.DatasetClasses.BaseNormalityDataset import BaseNormalityDataset


class NormalityTurbofanDataset(BaseNormalityDataset):
    def __init__(self,
                 data_dir_path: str,
                 dataset_csv_name: str,
                 window_length: int,
                 norm_op_len: int,
                 norm_op_as_pct: bool = False,
                 normalize: bool = False,
                 DEBUGGING: bool = False):

        # Define the column in the DataFrame that identifies every unit in the dataset
        unit_identifier_col_name = "unit"
        feature_names = ["cycle",
                        "operational_setting_1", "operational_setting_2", "operational_setting_3",
                         "sensor_measurement_1", "sensor_measurement_2", "sensor_measurement_3",
                         "sensor_measurement_4", "sensor_measurement_5", "sensor_measurement_6",
                         "sensor_measurement_7", "sensor_measurement_8", "sensor_measurement_9",
                         "sensor_measurement_10", "sensor_measurement_11", "sensor_measurement_12",
                         "sensor_measurement_13", "sensor_measurement_14", "sensor_measurement_15",
                         "sensor_measurement_16", "sensor_measurement_17", "sensor_measurement_18",
                         "sensor_measurement_19", "sensor_measurement_20", "sensor_measurement_21"]

        super(NormalityTurbofanDataset, self).__init__(data_dir_path=data_dir_path,
                                                       dataset_csv_name=dataset_csv_name,
                                                       window_length=window_length,
                                                       norm_op_len=norm_op_len,
                                                       unit_identifier_col_name=unit_identifier_col_name,
                                                       feature_names=feature_names,
                                                       norm_op_as_pct=norm_op_as_pct,
                                                       normalize=normalize,
                                                       DEBUGGING=DEBUGGING)

    def __getitem__(self, idx):
        # Every sample in the data array should have the following structure:
        # [
        #   "unit","cycle","operational_setting_1","operational_setting_2","operational_setting_3",
        #   "sensor_measurement_1", "sensor_measurement_2","sensor_measurement_3","sensor_measurement_4",
        #   "sensor_measurement_5","sensor_measurement_6", "sensor_measurement_7","sensor_measurement_8",
        #   "sensor_measurement_9","sensor_measurement_10", "sensor_measurement_11", "sensor_measurement_12",
        #   "sensor_measurement_13","sensor_measurement_14", "sensor_measurement_15","sensor_measurement_16",
        #   "sensor_measurement_17","sensor_measurement_18", "sensor_measurement_19","sensor_measurement_20",
        #   "sensor_measurement_21","RUL"
        # ]
        # Where the first two columns represent the unit and cycle, respectively.
        window = np.array(self.data[idx], dtype=np.float32)

        # Create the features used for training and set as float32.
        # Also create the labels (RUL) which include system name for labeling purposes.
        # Features are all columns except the first two (Unit and cycle) and last one (RUL),
        # RUL = last column
        # TODO: Perhaps also return the 'cycle' as a feature?
        unit = window[:, 0][0].astype(int)
        # features = window[:, 2:-1]
        features = window[:, 1:-1]
        rul = window[:, -1][-1]

        # Return features as target because we want to reconstruct the features themselves
        # [unit, features, target]
        # return [unit, features, features]
        return [features, features]


# =============================================================== #
# The code below is used to test the implemented dataset.         #
# class above.                                                    #
# =============================================================== #
def __test__():
    import math
    from torch.utils.data import DataLoader
    from torch.utils.data import random_split

    root_data_dir_path = ("/Users/rafaeltoche/Documents/School/Research/Rainwaters_Lab/"
                          "DART-LP2/Condition_Monitoring/data/NASA_turbofan_data/train/")
    csv_dataset_name = "FD002_train.csv"
    window_length = 1
    normal_op_pct = 30
    print("Reading data from: \n{}".format(os.path.join(root_data_dir_path, csv_dataset_name)))


    NASA_dataset = NormalityTurbofanDataset(data_dir_path=root_data_dir_path,
                                            dataset_csv_name=csv_dataset_name,
                                            window_length=window_length,
                                            norm_op_len=normal_op_pct,
                                            norm_op_as_pct=False,
                                            normalize=True,
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
