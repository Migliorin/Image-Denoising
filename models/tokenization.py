import torch.nn as nn
from einops.layers.torch import Rearrange


class UnPatchTokenization(nn.Module):
    def __init__(self,patch_size=50, img_size=224):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.h = self.img_size//self.patch_size
        self.w = self.img_size//self.patch_size
        self.unpatch_token = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=self.patch_size,
                                       p2=self.patch_size,h=self.h,w=self.w)

    def forward(self,x):
        return self.unpatch_token(x)

class PatchTokenization(nn.Module):
    def __init__(self, patch_size=50, token_len=768, channels=3):
        """Patch Tokenization Module
        
        Args:
            patch_size (int): the side length of a square patch
            token_len (int): desired length of an output token
            channels (int): channel of input image
        """
        super().__init__()
        self.patch_size = patch_size
        self.token_len = token_len
        self.channels = channels

        # Layer to split the image into patches
        self.split = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        
        # Linear layer to project patches to the token length
        self.project = nn.Linear((self.patch_size**2) * self.channels, token_len)

    def forward(self, x):
        """Forward pass of the PatchTokenization module.
        
        Args:
            x (torch.Tensor): Input image tensor
            
        Returns:
            torch.Tensor: Encoded image tensor
        """
        # Split image into patches and rearrange the dimensions
        x = self.split(x)
        
        # Project the patches to the desired token length
        x = self.project(x)
        return x