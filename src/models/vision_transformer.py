import numpy as np
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import repeat
from util import get_sinusoid_encoding
from models.tokenization import PatchTokenizationEinops, UnPatchTokenizationEinops


class VisionModelVisionTransformer(nn.Module):
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
        super(VisionModelVisionTransformer, self).__init__()
        
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
        self.patch_tokenization = PatchTokenizationEinops(
            patch_size=patch_size,
            token_len=token_len,
            channels=C
        )

        self.unpatch_tokenization = UnPatchTokenizationEinops(
            patch_size=patch_size,
            img_size=H
        )
        
        self.transformer = nn.Transformer(embed_dim, num_heads, num_layers)

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, self.token_len),
            requires_grad=True
        )
        
        self.emb_posi = nn.Parameter(
            data=get_sinusoid_encoding(self.num_tokens + 1, self.token_len),
            requires_grad=False
        )

        self.decoder_emb_posi = nn.Parameter(
            data=get_sinusoid_encoding(self.num_tokens, self.token_len),
            requires_grad=False
        )
        

        self.linear = nn.Linear(self.token_len, (self.patch_size**2) * C)

        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.2)
        


    def forward(self,x):
        # Apply patch tokenization to the input image
        x = self.patch_tokenization(x)
        
        B, N, E = x.shape
        
        # Add positional encoding to the tokens
        encoder = x + self.emb_posi[:,1:,:]
        decoder = x + self.decoder_emb_posi

        cls_token = self.cls_token + self.emb_posi[:,:1,:]
        
        # Expand the class token to the batch size and concatenate with image tokens
        cls_token = cls_token.expand(B, -1, -1)
        
        encoder = torch.cat((cls_token, encoder), dim=1)
        decoder = torch.cat((cls_token, decoder),dim=1)
        
        x = self.transformer(encoder, decoder)
        
        x = self.dropout(x)

        x = self.linear(x)

        x = self.sigmoid(x)

        return x
