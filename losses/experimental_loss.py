from einops.layers.torch import Rearrange
import torch.nn.functional as F
import torch.nn as nn
import torch



class RMSELossPatch(nn.MSELoss):
    def __init__(self,patch_size: int):
        super(RMSELossPatch,self).__init__()
        self.rea1 = Rearrange('b c (p1 h) (p2 w) -> b c (p1 p2) (h w)', p1=patch_size, p2=patch_size)


    def forward(self, input_image:torch.tensor, target_image: torch.tensor):
        input_image = self.rea1(input_image)
        target_image = self.rea1(target_image)

        tensor_loss = F.mse_loss(input_image,target_image,reduction='none')
        tensor_loss = torch.mean(tensor_loss,dim=-1)
        tensor_loss = torch.sqrt(tensor_loss.sum())

        return tensor_loss

