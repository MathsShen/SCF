import torch
import torch.nn as nn
from ive import *
import math
import pdb


class KLDiracVMF(nn.Module):
    def __init__(self, z_dim, radius):
        super().__init__()
        self.z_dim = z_dim
        self.radius = radius
        

    def forward(self, mu, kappa, wc):
        # mu and wc: (B, dim)
        # kappa: (B, 1)

        B = mu.size(0)
        d = self.z_dim
        r = self.radius

        log_ive_kappa = torch.log(1e-6 + ive(d/2-1, kappa))
        log_iv_kappa = log_ive_kappa + kappa

        cos_theta = torch.sum(mu * wc, dim=1, keepdim=True) / r

        l1 = -kappa * cos_theta
        l2 = - (d/2-1) * torch.log(1e-6 + kappa)
        l3 = log_iv_kappa * 1.0

        losses = l1 + l2 + l3 \
                + (d/2) * math.log(2*math.pi) \
                + d * math.log(r)

        return losses, l1, l2, l3


