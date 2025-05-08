import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class TokenizerImage(nn.Module):
    def __init__(self, d_patch: int, d_model: int, patch_size: int):
        super(TokenizerImage, self).__init__()
        self.rea1 = Rearrange(
            'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
        self.linear = nn.Linear(d_patch, d_model)
#        self.positional_encoder = PositionalEncoding(seq_length, d_model)

    def forward(self, x: torch.tensor):
        x = self.rea1(x)
        x = self.linear(x)

#        x = self.positional_encoder(x)

        return x


class UnTokenizerImage(nn.Module):
    def __init__(self, seq_length: int, patch_size: int, channel: int, width: int,
                 height: int):
        super(UnTokenizerImage, self).__init__()
        self.rea1 = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=patch_size,
                              p2=patch_size, h=height, w=width, c=channel)
        # self.linear = nn.Linear(d_model, seq_length)

    def forward(self, x):
        # x = self.linear(x)
        x = self.rea1(x)

        return x


class PositionalEncoding(nn.Module):
    """
    Implements the positional encoding layer for Transformer models.
    This adds information about the relative or absolute position of tokens in the sequence
    using sine and cosine functions of different frequencies.

    The positional encoding has the same dimension (d_model) as the input embeddings so that
    they can be summed together.

    Reference: "Attention Is All You Need" (Vaswani et al., 2017)
    """

    def __init__(self, seq_length: int, d_model: int, n: int = 10000, dtype: torch.dtype = torch.float32):
        """
        Initialize the Positional Encoding layer.

        Args:
            seq_length (int): Maximum length of input sequences (used to pre-compute encoding table)
            d_model (int): Dimension of the model embeddings (must be even)
            n (int, optional): Scaling factor for frequency calculation. Default: 10000
            dtype (torch.dtype, optional): Data type for encoding table. Default: torch.float32
        """
        super(PositionalEncoding, self).__init__()
        self.seq_length = seq_length
        self.d_model = d_model
        self.n = n
        self.dtype = dtype

        # Pre-compute the encoding table during initialization
        self.encode_table = self._create_table()

    def _create_table(self) -> torch.tensor:
        """
        Create the positional encoding table.

        Returns:
            torch.tensor: Encoding table of shape (seq_length, d_model)
                         containing positional encodings for all positions
        """
        # Initialize table with zeros
        table = torch.zeros((self.seq_length, self.d_model), dtype=self.dtype)

        # For each position in the sequence
        for pos in torch.arange(self.seq_length):
            # For each dimension (using half because we process pairs)
            for i in torch.arange(self.d_model // 2):
                # Calculate the denominator for frequency scaling
                denominator = 2 * i / self.d_model
                # Calculate the position-dependent value
                calculation = pos / torch.pow(self.n, denominator)

                # Apply sine to even indices and cosine to odd indices
                table[pos, 2 * i] = torch.sin(calculation)     # Even indices
                table[pos, 2 * i + 1] = torch.cos(calculation)  # Odd indices

        return table

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass adds positional encoding to input tensor.

        Args:
            x (torch.tensor): Input tensor of shape (batch_size, seq_length, d_model)

        Returns:
            torch.tensor: Output tensor with positional encoding added (same shape as input)
        """
        # Add positional encoding to input (broadcasting over batch dimension)
        x += self.encode_table  # Only use relevant portion of encoding table

        return x
