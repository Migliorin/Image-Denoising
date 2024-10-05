import torch.nn.functional as F
from torch import nn
import torch


class MSELossPatchEinops(nn.MSELoss):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

        self.split_token = Rearrange(
            'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
            p1=self.patch_size, 
            p2=self.patch_size
        )

    def forward(self,input_,target_):
        target_ = self.split_token(target_)
        tensor = F.mse_loss(input_,target_,reduction='none')
        tensor = tensor.sum(dim=-1).sum()
        return tensor


