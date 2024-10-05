import torch.nn.functional as F
from torch import nn
import torch


class MSELossPatch(nn.MSELoss):
    def __init__(self):
        super().__init__()


    def forward(self,input_,target_):
        tensor = F.mse_loss(input_,target_,reduction='none')
        tensor = tensor.sum(dim=-1).sum()
        return tensor

