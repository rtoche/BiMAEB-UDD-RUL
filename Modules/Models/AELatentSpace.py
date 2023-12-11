from torch import nn
from torch import relu


class AELatentSpace(nn.Module):
    def __init__(self, input_size, latent_size):
        super(AELatentSpace, self).__init__()
        # This layer will output the means and variance. This has to be an even number as
        # half pertain to the means and half to the variance
        self.latent_space = nn.Linear(in_features=input_size, out_features=latent_size)

    def forward(self, x):
        return self.latent_space(x)
