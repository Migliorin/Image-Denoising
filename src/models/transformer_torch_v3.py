import numpy as np
import torch
import torch.nn as nn
from util import get_sinusoid_encoding
from models.tokenization import PatchTokenizationEinops

class VisionModelTransformerTorchV3(nn.Module):
    def __init__(self, img_size, patch_size, token_len, embed_dim=512, num_heads=8, num_layers=6):
        super(VisionModelTransformerTorchV3, self).__init__()
        
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
        
        # Define the transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                   nhead=num_heads,norm_first=True,batch_first=True)
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
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim,
                                                   nhead=num_heads,norm_first=True,batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Define the linear layers for image reconstruction
        self.linear = nn.Linear(self.token_len, (self.patch_size**2) * C)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply patch tokenization to the input image
        image_token = self.patch_tokenization(x)
        
        B, N, E = image_token.shape
        
        # Expand the class token to the batch size and concatenate with image tokens
        tokens = self.cls_token.expand(B, -1, -1)
        image_token = torch.cat((tokens, image_token), dim=1)
        
        # Add positional encoding to the tokens
        image_token += self.emb_posi
        
        # Permute dimensions to match transformer input requirements
        #image_token = image_token.permute(1, 0, 2)

        # Pass tokens through the transformer encoder
        out_encode = self.transformer_encoder(image_token)

        # Prepare decoder input (CLS token output from the encoder)
        decoder_input = out_encode[:, :1, :]

        # Pass the encoder output and decoder input through the transformer decoder
        decoder_output = self.transformer_decoder(out_encode[:,1:,:], decoder_input)

        # Pass the decoder output through the linear layers to reconstruct the image
        #x = decoder_output
        x = self.linear(decoder_output)
        x = self.sigmoid(x)

        return x
