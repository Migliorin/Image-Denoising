import numpy as np
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import repeat



class UnPatchTokenization(nn.Module):
    def __init__(self,patch_size=50, img_size=(3,224,224)):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.h = self.img_size[1]//self.patch_size
        self.w = self.img_size[2]//self.patch_size
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




class VisionModel(nn.Module):
    def __init__(self, img_size, patch_size, token_len, embed_dim=512, num_heads=8, num_layers=6):
        """Vision Transformer Model
        
        Args:
            img_size (tuple): Size of input image (B, C, H, W)
            patch_size (int): Size of the square patches
            token_len (int): Desired length of an output token
            embed_dim (int): Dimensionality of the token embeddings
            num_heads (int): Number of attention heads in the transformer layers
            num_layers (int): Number of transformer layers
        """
        super(VisionModel, self).__init__()
        
        self.img_size = img_size
        B, C, H, W = self.img_size
        self.patch_size = patch_size
        self.token_len = token_len
        
        # Ensure the height and width are divisible by the patch size
        assert H % self.patch_size == 0, 'Height of image must be evenly divisible by patch size.'
        assert W % self.patch_size == 0, 'Width of image must be evenly divisible by patch size.'
        
        # Calculate the number of tokens
        self.num_tokens = (H // self.patch_size) * (W // self.patch_size)
        
        # Initialize the patch tokenization module
        self.patch_tokenization = PatchTokenization(
            patch_size=patch_size,
            token_len=token_len,
            channels=C
        )

        self.unpatch_tokenization = UnPatchTokenization(
            patch_size=patch_size,
            img_size=(C,H,W)
        )

        # Define the transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Define CLS_token
        self.cls_token = nn.Parameter(torch.rand(1, 1, self.token_len))
        # Positional encoding
        self.emb_posi = nn.Parameter(
            data=torch.rand(1,self.num_tokens + 1, self.token_len))
        # Define the transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Define the linear layers for image reconstruction
        self.linear = nn.Linear(self.token_len, (self.patch_size**2) * C)
        self.sigmoid = nn.Sigmoid()
        self.final_layer = nn.Linear((self.patch_size**2) * C, C * H * W)


    def forward(self,x):
        image_token = self.patch_tokenization(x)
        B, N, _ = image_token.shape

        cls_tokens = repeat(self.cls_token,'1 1 d -> b 1 d',b = B)
        x = torch.cat([cls_tokens,image_token],dim=1)

        x += self.emb_posi[:,:(N + 1)]

        # Pass tokens through the transformer encoder
        out_encode = self.transformer_encoder(x)

        cls_tokens_output = repeat(out_encode[:,:1,:],'1 1 d -> b n d',b = B, n = N)

        output_decoder = self.transformer_decoder(image_token,cls_tokens_output)

        output_decoder = self.linear(output_decoder)

        output_decoder = self.sigmoid(output_decoder)

        output_decoder = self.unpatch_tokenization(output_decoder)

        return output_decoder

