import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import PositionalEncoding


def scaled_dot_product_attention(query: torch.tensor,
                                 key: torch.tensor,
                                 value: torch.tensor,
                                 mask: torch.tensor = None) -> tuple[torch.tensor, torch.tensor]:
    """
    Computes scaled dot-product attention as described in "Attention Is All You Need".

    Args:
        query (torch.tensor): Query tensor
        key (torch.tensor): Key tensor
        value (torch.tensor): Value tensor of shape
        mask (torch.tensor, optional): Mask tensor

    Returns:
        tuple[torch.tensor, torch.tensor]:
            - Output tensor
            - Attention weights tensor
    """
    # Calculate scaling factor to prevent dot products from growing too large
    # Scaling by 1/sqrt(d_k) as per the Transformer paper
    factor = 1 / torch.sqrt(torch.tensor(key.size(-1)))

    # Compute raw attention scores (dot products between queries and keys)
    attn = torch.matmul(query, key.transpose(-2, -1)) * factor

    # Apply mask if provided (sets masked positions to -inf before softmax)
    if mask is not None:
        attn = attn.masked_fill(mask == 0, -torch.inf)

    # Apply softmax to get normalized attention weights (sum to 1 along last dimension)
    attn = F.softmax(attn, dim=-1)

    # Multiply attention weights with values to get final output
    x = torch.matmul(attn, value)

    return x, attn


class HeadAttention(nn.Module):
    """
    Implements a single attention head as used in the Transformer architecture.
    This head performs linear transformations of queries, keys, and values,
    followed by scaled dot-product attention.

    Reference: "Attention Is All You Need" (Vaswani et al., 2017)
    """

    def __init__(self, d_model: int, d_k: int, d_v: int):
        """
        Initialize the attention head with learnable weight matrices.

        Args:
            d_model (int): Dimension of input embeddings (model dimension)
            d_k (int): Dimension of key and query projections
            d_v (int): Dimension of value projections
        """
        super(HeadAttention, self).__init__()
        # Learnable weight matrix for query projection
        self.weights_query = nn.Parameter(torch.randn(d_model, d_k))
        # Learnable weight matrix for key projection
        self.weights_key = nn.Parameter(torch.randn(d_model, d_k))
        # Learnable weight matrix for value projection
        self.weights_value = nn.Parameter(torch.randn(d_model, d_v))

    def forward(self, query: torch.tensor, key: torch.tensor,
                value: torch.tensor, mask: torch.tensor = None) -> torch.tensor:
        """
        Forward pass for the attention head.

        Args:
            query (torch.tensor): Query tensor
            key (torch.tensor): Key tensor
            value (torch.tensor): Value tensor
            mask (torch.tensor, optional): Attention mask

        Returns:
            torch.tensor: Output tensor
        """
        # Project inputs to lower dimensional spaces
        q = torch.matmul(query, self.weights_query)
        k = torch.matmul(key, self.weights_key)
        v = torch.matmul(value, self.weights_value)

        # Apply scaled dot-product attention
        x, _ = scaled_dot_product_attention(q, k, v, mask=mask)

        return x


class MultiHead(nn.Module):
    """
    Implements multi-head attention as described in "Attention Is All You Need".
    This consists of multiple attention heads running in parallel, whose outputs
    are concatenated and linearly transformed, followed by residual connection
    and layer normalization.
    """

    def __init__(self, d_model: int, d_k: int, d_v: int, h: int):
        """
        Initialize the multi-head attention layer.

        Args:
            d_model (int): Dimension of input and output embeddings
            d_k (int): Dimension of key and query projections for each head
            d_v (int): Dimension of value projections for each head
            h (int): Number of parallel attention heads
        """
        super(MultiHead, self).__init__()

        # Linear transformation for concatenated head outputs
        self.weights_concat = nn.Parameter(torch.randn(d_v * h, d_model))

        # Create h parallel attention heads
        self.multi_head = nn.ModuleList([
            HeadAttention(d_model=d_model, d_k=d_k, d_v=d_v)
            for _ in range(h)
        ])

        # Layer normalization for the output
        self.norm_layer = nn.LayerNorm(d_model)

    def forward(self, query: torch.tensor, key: torch.tensor,
                value: torch.tensor, mask: torch.tensor = None) -> torch.tensor:
        """
        Forward pass for multi-head attention.

        Args:
            query (torch.tensor): Query tensor
            key (torch.tensor): Key tensor
            value (torch.tensor): Value tensor
            mask (torch.tensor, optional): Attention mask

        Returns:
            torch.tensor: Output tensor

        Note:
            - Implements residual connection (Add & Norm)
            - Output dimension matches input dimension (d_model)
        """
        # Process input through all attention heads in parallel
        x = torch.cat([head(query, key, value, mask=mask)
                      for head in self.multi_head], dim=-1)

        # Linearly transform the concatenated outputs
        x = torch.matmul(x, self.weights_concat)

        # Add residual connection (original query)
        x += query

        # Apply layer normalization
        x = self.norm_layer(x)

        return x


class PositionWiseFeedForwardNetworks(nn.Module):
    """
    Implements the position-wise feed-forward network (FFN) used in Transformer models.
    This consists of two linear transformations with a ReLU activation in between,
    followed by a residual connection and layer normalization.

    Reference: "Attention Is All You Need" (Vaswani et al., 2017)
    """

    def __init__(self, d_model: int, d_ff: int):
        """
        Initialize the position-wise feed-forward network.

        Args:
            d_model (int): Dimension of input and output embeddings
            d_ff (int): Dimension of inner layer (typically 4*d_model)

        Note:
            - The FFN is applied to each position separately and identically
            - Original paper uses d_ff = 2048 when d_model = 512
        """
        super(PositionWiseFeedForwardNetworks, self).__init__()

        # First linear transformation (expands dimension)
        self.linear_inner_layer = nn.Linear(d_model, d_ff)

        # Second linear transformation (projects back to original dimension)
        self.linear_layer = nn.Linear(d_ff, d_model)

        # Layer normalization for the output
        self.norm_layer = nn.LayerNorm(d_model)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass for the position-wise feed-forward network.

        Args:
            x (torch.tensor): Input tensor

        Returns:
            torch.tensor: Output tensor
       """

        # Save original input for residual connection
        residual = x

        # First linear transformation + ReLU activation
        x = self.linear_inner_layer(x)
        x = F.relu(x)

        # Second linear transformation
        x = self.linear_layer(x)

        # Add residual connection (original input)
        x += residual

        # Apply layer normalization
        x = self.norm_layer(x)

        return x


class EncoderLayer(nn.Module):
    """
    Implements a single encoder layer for the Transformer architecture.
    Each encoder layer consists of:
    1. Multi-head self-attention mechanism
    2. Position-wise feed-forward network
    Both sub-layers employ residual connections followed by layer normalization.

    Reference: "Attention Is All You Need" (Vaswani et al., 2017)
    """

    def __init__(self, d_model: int, d_k: int, d_v: int, h: int, d_ff: int):
        """
        Initialize the encoder layer components.

        Args:
            d_model (int): Dimension of input and output embeddings
            d_k (int): Dimension of key and query projections for each head
            d_v (int): Dimension of value projections for each head
            h (int): Number of parallel attention heads
            d_ff (int): Inner dimension of the feed-forward network
        """
        super(EncoderLayer, self).__init__()

        # Multi-head self-attention mechanism
        self.multi_head_attention = MultiHead(
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            h=h
        )

        # Position-wise feed-forward network
        self.position_wise_feed_forward_networks = PositionWiseFeedForwardNetworks(
            d_model=d_model,
            d_ff=d_ff
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass for the encoder layer.

        Args:
            x (torch.tensor): Input tensor

        Returns:
            torch.tensor: Output tensor

        Note:
            - For self-attention, queries, keys and values all come from the same input
        """
        # Self-attention (query, key, value all come from the same input)
        # The mask is applied to prevent attention to certain positions
        x = self.multi_head_attention(
            query=x,
            key=x,
            value=x
        )

        # Position-wise feed-forward network
        x = self.position_wise_feed_forward_networks(x)

        return x


class Encoder(nn.Module):
    """
    Implements the complete encoder stack for the Transformer architecture.
    The encoder consists of:
    1. Input embedding with positional encoding
    2. A stack of N identical encoder layers

    Reference: "Attention Is All You Need" (Vaswani et al., 2017)
    """

    def __init__(self, seq_length: int, n_layers: int, d_model: int,
                 d_k: int, d_v: int, h: int, d_ff: int):
        """
        Initialize the Transformer encoder.

        Args:
            seq_length (int): Maximum sequence length for positional encoding
            n_layers (int): Number of identical encoder layers in the stack
            d_model (int): Dimension of input embeddings and hidden states
            d_k (int): Dimension of key and query projections for each head
            d_v (int): Dimension of value projections for each head
            h (int): Number of attention heads in each layer
            d_ff (int): Inner dimension of the position-wise feed-forward networks

        Note:
            - Typical configuration for base model:
              d_model=512, h=8, d_k=d_v=64, d_ff=2048
        """
        super(Encoder, self).__init__()

        # Positional encoding layer
        self.positional_encoding = PositionalEncoding(
            seq_length=seq_length,
            d_model=d_model
        )

        # Stack of encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                d_k=d_k,
                d_v=d_v,
                h=h,
                d_ff=d_ff
            ) for _ in range(n_layers)
        ])

    def forward(self, src: torch.tensor) -> torch.tensor:
        """
        Forward pass for the Transformer encoder.

        Args:
            src (torch.tensor): Input sequence tensor of shape

        Returns:
            torch.tensor: Encoded output tensor
        """

        # Add positional information to input embeddings
        src = self.positional_encoding(src.cpu())
        src = src.cuda()

        # Process through each encoder layer
        for module_ in self.encoder_layers:
            # Each layer handles its own residual connections
            src = module_(src)

        return src


class DecoderLayer(nn.Module):
    """
    Implements a single decoder layer for the Transformer architecture.
    Each decoder layer consists of:
    1. Masked multi-head self-attention (prevents looking at future tokens)
    2. Multi-head attention over encoder outputs (memory)
    3. Position-wise feed-forward network
    All sub-layers include residual connections and layer normalization.

    Reference: "Attention Is All You Need" (Vaswani et al., 2017)
    """

    def __init__(self, d_model: int, d_k: int, d_v: int, h: int, d_ff: int):
        """
        Initialize the decoder layer components.

        Args:
            d_model (int): Dimension of input and output embeddings
            d_k (int): Dimension of key and query projections for each head
            d_v (int): Dimension of value projections for each head
            h (int): Number of parallel attention heads
            d_ff (int): Inner dimension of the feed-forward network
        """
        super(DecoderLayer, self).__init__()

        # Masked multi-head self-attention (prevents looking ahead)
        self.masked_multi_head_attention = MultiHead(
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            h=h
        )

        # Multi-head attention over encoder outputs
        self.multi_head_attention_memory = MultiHead(
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            h=h
        )

        # Position-wise feed-forward network
        self.position_wise_feed_forward_networks = PositionWiseFeedForwardNetworks(
            d_model=d_model,
            d_ff=d_ff
        )

    def forward(self, x: torch.tensor, memory: torch.tensor, tgt_mask: torch.tensor) -> torch.tensor:
        """
        Forward pass for the decoder layer.

        Args:
            x (torch.tensor): Target sequence tensor
            memory (torch.tensor): Encoder output tensor
            tgt_mask (torch.tensor): Target mask

        Returns:
            torch.tensor: Output tensor
        """

        # Masked self-attention (query, key, value all from target sequence)
        x = self.masked_multi_head_attention(
            query=x,
            key=x,
            value=x,
            mask=tgt_mask  # Optional
        )

        # Attention over encoder memory (query from decoder, key/value from encoder)
        x = self.multi_head_attention_memory(
            query=x,
            key=memory,
            value=memory
        )

        # Position-wise feed-forward network
        x = self.position_wise_feed_forward_networks(x)

        return x


class Decoder(nn.Module):
    """
    Implements the complete decoder stack for the Transformer architecture.
    The decoder consists of:
    1. Target embedding with positional encoding
    2. A stack of N identical decoder layers
    3. Each layer processes both target sequence and encoder memory

    Reference: "Attention Is All You Need" (Vaswani et al., 2017)
    """

    def __init__(self, seq_length: int, n_layers: int, d_model: int,
                 d_k: int, d_v: int, h: int, d_ff: int):
        """
        Initialize the Transformer decoder.

        Args:
            seq_length (int): Maximum sequence length for positional encoding
            n_layers (int): Number of identical decoder layers in the stack
            d_model (int): Dimension of input embeddings and hidden states
            d_k (int): Dimension of key and query projections for each head
            d_v (int): Dimension of value projections for each head
            h (int): Number of attention heads in each layer
            d_ff (int): Inner dimension of the position-wise feed-forward networks
        """
        super(Decoder, self).__init__()

        # Positional encoding layer
        self.positional_encoding = PositionalEncoding(
            seq_length=seq_length,
            d_model=d_model
        )

        # Stack of decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(
                d_model=d_model,
                d_k=d_k,
                d_v=d_v,
                h=h,
                d_ff=d_ff
            ) for _ in range(n_layers)
        ])

    def forward(self, tgt: torch.tensor, memory: torch.tensor, tgt_mask: torch.tensor) -> torch.tensor:
        """
        Forward pass for the Transformer decoder.

        Args:
            tgt (torch.tensor): Target sequence tensor
            memory (torch.tensor): Encoder output tensor
            tgt_mask (torch.tensor): Mask for decoder

        Returns:
            torch.tensor: Decoded output tensor
        """

        # Add positional information to target embeddings
        tgt = self.positional_encoding(tgt.cpu())
        tgt = tgt.cuda()

        # Process through each decoder layer
        for module_ in self.decoder_layers:
            # Each layer handles its own residual connections
            tgt = module_(
                x=tgt,
                memory=memory,
                tgt_mask=tgt_mask
            )

        return tgt


class TransformerFromScratch(nn.Module):
    """
    Complete Transformer model implementation from scratch.
    Consists of encoder-decoder architecture with:
    - Multi-head attention mechanisms
    - Position-wise feed-forward networks
    - Positional encoding
    - Final linear projection and softmax

    Reference: "Attention Is All You Need" (Vaswani et al., 2017)
    """

    def __init__(self, seq_length: int, n_encoder_layers: int, n_decoder_layers: int,
                 progession: int, d_model: int, d_k: int, d_v: int, h: int, d_ff: int):
        """
        Initialize the complete Transformer model.

        Args:
            seq_length (int): Maximum sequence length for positional encoding
            n_encoder_layers (int): Number of layers in encoder stack
            n_decoder_layers (int): Number of layers in decoder stack
            progession (int): Output dimension for final linear layer
            d_model (int): Dimension of embeddings and hidden states
            d_k (int): Dimension of key/query projections per head
            d_v (int): Dimension of value projections per head
            h (int): Number of attention heads
            d_ff (int): Inner dimension of position-wise feed-forward networks

        Note:
            - Typical base model: d_model=512, h=8, d_k=d_v=64, d_ff=2048
        """
        super(TransformerFromScratch, self).__init__()

        # Encoder stack
        self.encoder = Encoder(
            seq_length=seq_length,
            n_layers=n_encoder_layers,
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            h=h,
            d_ff=d_ff
        )

        # Decoder stack
        self.decoder = Decoder(
            seq_length=seq_length,
            n_layers=n_decoder_layers,
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            h=h,
            d_ff=d_ff
        )

        # Final linear projection layer
        self.linear = nn.Parameter(torch.randn((d_model, progession)))

    def forward(self, src: torch.tensor, tgt: torch.tensor,
                tgt_mask: torch.tensor) -> torch.tensor:
        """
        Forward pass for complete Transformer model.

        Args:
            src (torch.tensor): Source sequence tensor
            tgt (torch.tensor): Target sequence tensor
            tgt_mask (torch.tensor): Look-ahead mask for decoder
        Returns:
            torch.tensor: Output probabilities
        """
        # Encode source sequence
        memory = self.encoder(src=src)

        # Decode target sequence using encoder memory
        x = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask
        )

        # Project to output dimension
        x = torch.matmul(x, self.linear)

        return x
