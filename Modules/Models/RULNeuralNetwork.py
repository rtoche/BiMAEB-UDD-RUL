import torch
from torch import nn
from Modules.Models.BaseModel import BaseModel


class RULNeuralNetwork(BaseModel):
    def __init__(self,
                 ts_number_features: int,
                 device: str,
                 model_name: str,
                 metrics_dir: str):

        super(RULNeuralNetwork, self).__init__(model_name, metrics_dir)

        self.number_features = ts_number_features
        self.device = device

        fc1_size = 128
        fc2_size = 128
        fc3_size = 128
        fc4_size = 128

        self.nn = nn.Sequential(
            nn.Linear(in_features=ts_number_features, out_features=fc1_size),
            nn.ReLU(),
            nn.Linear(in_features=fc1_size, out_features=fc2_size),
            nn.ReLU(),
            nn.Linear(in_features=fc2_size, out_features=fc3_size),
            nn.ReLU(),
            nn.Linear(in_features=fc3_size, out_features=fc4_size),
            nn.ReLU(),
            nn.Linear(in_features=fc4_size, out_features=1),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.nn(x)
        return out

    def __do_batch__(self, batch: int, loss_function, device: str):
        features, targets = batch

        targets = torch.unsqueeze(targets, dim=1)
        # Move everything to the device
        features = features.to(device)
        targets = targets.to(device)

        predictions = self(features)
        predictions = predictions.to(device)
        loss = loss_function(predictions, targets)

        return loss
