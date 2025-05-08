import torch
import torch.nn as nn
import torch.functional as F
from tokenizer import TokenizerImage, UnTokenizerImage
from models import TransformerFromScratch


class ModuleOne(nn.Module):
    def __init__(self, seq_length: int, d_patch: int, d_model: int, progession: int, patch_size: int, n_encoder_layers: int, n_decoder_layers: int, d_k: int, d_v: int, h: int, d_ff: int):
        super(ModuleOne, self).__init__()
        self.tokenize_image = TokenizerImage(
            patch_size=patch_size,
            d_model=d_model,
            d_patch=d_patch
        )

        self.transformer = TransformerFromScratch(
            seq_length=seq_length,
            n_encoder_layers=n_encoder_layers,
            n_decoder_layers=n_decoder_layers,
            progession=progession,
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            h=h,
            d_ff=d_ff
        )

        self.relu = nn.ReLU()

        self.inception = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=(1, 3), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=(3, 3), padding=(0, 2)),
            nn.BatchNorm2d(1),
            nn.Flatten(start_dim=1, end_dim=2)
        )
        self.untokenizer_images = UnTokenizerImage(
            seq_length=seq_length,
            patch_size=patch_size,
            channel=3,
            width=32,
            height=32
        )
        
        self.mask = nn.Parameter(torch.triu(torch.ones(1024, 1024)).T,requires_grad=False)

    def forward(self, b_image: torch.tensor, mask: torch.tensor = None) -> torch.tensor:
        tokenized_images = self.tokenize_image(b_image)
        rearragened_images = self.tokenize_image.rea1(b_image)
        
        if(mask is None):
            mask = self.mask

        x = self.transformer(tokenized_images, tokenized_images, mask)
        x = self.relu(x)

        x = torch.concat([
            rearragened_images.unsqueeze(1),
            x.unsqueeze(1),
        ], dim=1)

        x = self.inception(x)

        x = self.untokenizer_images(x)

        return x
