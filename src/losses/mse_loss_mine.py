import torch.nn.functional as F
from torch import nn
import torch

class MSELossMine(nn.MSELoss):
    def __init__(self):
        super().__init__()

    def forward(self,input_,target_):
        tensor = F.mse_loss(input_, target_, reduction='none')
        tensor = torch.mean(tensor, dim=1)
        tensor = torch.sum(tensor, dim=0).sum(dim=1).mean()
        return tensor

class MSELossPatch(nn.MSELoss):
    def __init__(self):
        super().__init__()

    def forward(self,input_,target_):
        tensor = F.mse_loss(input_,target_,reduction='none')
        tensor = tensor.sum(dim=1).mean()
        return tensor
