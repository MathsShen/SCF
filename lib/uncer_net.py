import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb


class UncertaintyNet(nn.Module):
    def __init__(self, convf_dim, z_dim):
        super().__init__()

        self.z_dim = z_dim
        self.convf_dim = convf_dim

        self._log_kappa = nn.Sequential(
                        nn.Linear(self.convf_dim, self.convf_dim // 2),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.convf_dim // 2, self.convf_dim // 4),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.convf_dim // 4, 1),
                        )


    def forward(self, convf):
        log_kappa = self._log_kappa(convf)
        log_kappa = torch.log(1e-6 + torch.exp(log_kappa))

        return log_kappa
