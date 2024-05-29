import torch
import torch.nn as nn
import numpy as np

# Define a class for patch tokenization
class PatchTokenization(nn.Module):
    def __init__(self, img_size=(1, 1, 60, 100), patch_size=50, token_len=768):
        super().__init__()
        
        # Initialize parameters
        self.img_size = img_size
        B, C, H, W = self.img_size
        self.patch_size = patch_size
        self.token_len = token_len
        # Ensure the image dimensions are divisible by the patch size
        assert H % self.patch_size == 0, 'Height of image must be evenly divisible by patch size.'
        assert W % self.patch_size == 0, 'Width of image must be evenly divisible by patch size.'
        self.num_tokens = (H // self.patch_size) * (W // self.patch_size)
        
        # Define layers
        self.split = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size, padding=0)
        self.project = nn.Linear((self.patch_size ** 2) * C, token_len)

    def forward(self, x):
        # Split image into patches and project each patch to a token
        x = self.split(x).transpose(2, 1)
        x = self.project(x)
        return x

# Define the vision model class
class VisionModel(nn.Module):
    def __init__(self, img_size, patch_size, token_len, embed_dim=512, num_heads=8, num_layers=6):
        super(VisionModel, self).__init__()
        
        # Initialize the patch tokenization module
        self.patch_tokenization = PatchTokenization(
            img_size=img_size,
            patch_size=patch_size,
            token_len=token_len
        )
        self.img_size = img_size
        B, C, H, W = self.img_size
        self.num_tokens = int(self.patch_tokenization.num_tokens)
        self.token_len = token_len
        self.patch_size = patch_size
        
        # Define the transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Define the class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.token_len))
        self.emb_posi = nn.Parameter(
            data=self.get_sinusoid_encoding(self.num_tokens + 1, self.token_len),
            requires_grad=False
        )

        # Define the transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Define the linear layer and fold operation for reconstructing the image
        self.linear = nn.Linear(self.token_len, (self.patch_size ** 2) * C)
        self.fold = nn.Fold(output_size=(H, W), kernel_size=self.patch_size, stride=self.patch_size)

    # Function to create sinusoidal position encoding
    def get_sinusoid_encoding(self, num_tokens, token_len):
        def get_position_angle_vec(i):
            return [i / np.power(10000, 2 * (j // 2) / token_len) for j in range(token_len)]
    
        sinusoid_table = np.array([get_position_angle_vec(i) for i in range(num_tokens)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) 
    
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)
    
    def forward(self, x):
        # Tokenize the input image into patches
        image_token = self.patch_tokenization(x)
        
        B, N, E = image_token.shape
        # Add class token to the patch tokens
        tokens = self.cls_token.expand(B, -1, -1)
        image_token = torch.cat((tokens, image_token), dim=1)
        image_token += self.emb_posi
        image_token = image_token.permute(1, 0, 2)

        # Pass the tokens through the transformer encoder
        out_encoder = self.transformer_encoder(image_token)

        # Pass the encoder output through the transformer decoder
        decoder_output = self.transformer_decoder(out_encoder[1:], out_encoder[0:1])
        
        # Project the decoded output back to the original image dimensions
        x = decoder_output.permute(1, 0, 2)
        x = self.linear(x)
        x = x.transpose(2, 1)
        x = self.fold(x)
        
        return x

