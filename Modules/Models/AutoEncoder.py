from torch import nn
from Modules.Models.BaseModel import BaseModel
from Modules.Models.AELatentSpace import AELatentSpace


class AutoEncoder(BaseModel):
    def __init__(self,
                 ts_number_features: int,
                 latent_size: int,
                 device: str,
                 model_name: str,
                 metrics_dir: str,
                 regularization: str = None,
                 regularization_weight = 0.005):

        super(AutoEncoder, self).__init__(model_name=model_name,
                                          metrics_dir=metrics_dir,
                                          regularization=regularization,
                                          regularization_weight=regularization_weight)

        self.model_type = "AutoEncoder"
        fc1_size = 512
        fc2_size = 256
        fc3_size = 128
        fc4_Size = 32

        self.encoder = nn.Sequential(
            nn.Linear(in_features=ts_number_features, out_features=fc1_size),
            nn.ReLU(),
            nn.Linear(in_features=fc1_size, out_features=fc2_size),
            nn.ReLU(),
            nn.Linear(in_features=fc2_size, out_features=fc3_size),
            nn.ReLU(),
            nn.Linear(in_features=fc3_size, out_features=fc4_Size),
            nn.ReLU()
        )

        self.latent_space = AELatentSpace(input_size=fc4_Size, latent_size=latent_size)

        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_features=latent_size, out_features=fc4_Size),
            nn.ReLU(),
            nn.Linear(in_features=fc4_Size, out_features=fc3_size),
            nn.ReLU(),
            nn.Linear(in_features=fc3_size, out_features=fc2_size),
            nn.ReLU(),
            nn.Linear(in_features=fc2_size, out_features=fc1_size),
            nn.ReLU(),
            nn.Linear(in_features=fc1_size, out_features=ts_number_features),
        )

    def forward(self, ts_data):
        assert ts_data.shape[1] == 1, "AutoEncoder expects inputs with a sequence of length 1."
        encoder_out = self.encoder(ts_data)
        latent_space = self.latent_space(encoder_out)
        decoder_out = self.decoder(latent_space)

        return decoder_out
