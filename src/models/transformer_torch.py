import numpy as np
import torch
import torch.nn as nn
from util import get_sinusoid_encoding
from models.tokenization import PatchTokenizationUnfold

class VisionModelTransformerTorch(nn.Module):
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
        super(VisionModelTransformerTorch, self).__init__()
        
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
        self.patch_tokenization = PatchTokenizationUnfold(
            patch_size=patch_size,
            token_len=token_len,
            channels=C
        )
        
        # Define the transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Define the class token and positional encoding
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, self.token_len),
            requires_grad=True
        )
        self.emb_posi = nn.Parameter(
            data=get_sinusoid_encoding(self.num_tokens + 1, self.token_len),
            requires_grad=True
        )

        # Define the transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Define the linear layers for image reconstruction
        self.linear = nn.Linear(self.token_len, (self.patch_size**2) * C)
        self.final_layer = nn.Linear((self.patch_size**2) * C, C * H * W)

    def forward(self, x):
        """Forward pass of the VisionModel.
        
        Args:
            x (torch.Tensor): Input image tensor
            
        Returns:
            torch.Tensor: Reconstructed image tensor
        """
        # Apply patch tokenization to the input image
        image_token = self.patch_tokenization(x)
        
        B, N, E = image_token.shape
        
        # Expand the class token to the batch size and concatenate with image tokens
        tokens = self.cls_token.expand(B, -1, -1)
        image_token = torch.cat((tokens, image_token), dim=1)
        
        # Add positional encoding to the tokens
        image_token += self.emb_posi
        
        # Permute dimensions to match transformer input requirements
        image_token = image_token.permute(1, 0, 2)

        # Pass tokens through the transformer encoder
        out_encode = self.transformer_encoder(image_token)

        # Prepare decoder input (CLS token output from the encoder)
        decoder_input = out_encode[0, :, :].unsqueeze(0)
        
        # Pass the encoder output and decoder input through the transformer decoder
        decoder_output = self.transformer_decoder(out_encode, decoder_input)
        
        # Pass the decoder output through the linear layers to reconstruct the image
        x = decoder_output[0, :, :]
        x = self.linear(x)
        x = self.final_layer(x)

        _, C, H, W = self.img_size
        
        # Reshape the output to the original image dimensions
        x = x.reshape(-1, C, H, W)
        
        return x

