import torch
import torch.nn as nn
from typing import Optional

from .attention import MultiHeadAttention, FeedForwardNN
from .encoder import PositionalEncoding
from seq2seq.data.fr_en import tokenizer


class DecoderLayer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embedding_dim: int,
        ffn_hidden_dim: int,
        qk_length: int,
        value_length: int,
        dropout: float = 0.1,
    ):
        """
        Each decoder layer will take in two embeddings of
        shape (B, T, C):

        1. The `target` embedding, which comes from the decoder
        2. The `source` embedding, which comes from the encoder

        and will output a representation
        of the same shape.

        The decoder layer will have three main components:
            1. A Masked Multi-Head Attention layer (you'll need to
               modify the MultiHeadAttention layer to handle this!)
            2. A Multi-Head Attention layer for cross-attention
               between the target and source embeddings.
            3. A Feed-Forward Neural Network layer.

        Remember that for each Multi-Head Attention layer, we
        need create Q, K, and V matrices from the input embedding(s)!
        """
        super().__init__()

        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.ffn_hidden_dim = ffn_hidden_dim
        self.qk_length = qk_length
        self.value_length = value_length

        self.dropout = nn.Dropout(dropout)
        self.self_attn = MultiHeadAttention(num_heads, embedding_dim, qk_length, value_length)
        self.cross_attn = MultiHeadAttention(num_heads, embedding_dim, qk_length, value_length)
        self.ffn = FeedForwardNN(embedding_dim, ffn_hidden_dim)

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)

        # Define any layers you'll need in the forward pass
        # raise NotImplementedError("Need to implement DecoderLayer layers")


    def forward(
        self,
        x: torch.Tensor,
        enc_x: torch.Tensor | None,
        tgt_mask: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        The forward pass of the DecoderLayer.
        """
        x = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(x))

        if enc_x is not None:
            x = self.cross_attn(x, enc_x, enc_x, mask=src_mask)
            x = self.norm2(x + self.dropout(x))

        x = self.ffn(x)
        x = self.norm3(x + self.dropout(x))

        return x
        raise NotImplementedError("Need to implement DecoderLayer forward pass.")


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        num_heads: int,
        embedding_dim: int,
        ffn_hidden_dim: int,
        qk_length: int,
        value_length: int,
        max_length: int,
        dropout: float = 0.1,
    ):
        """
        Remember that the decoder will take in a sequence
        of tokens AND a source embedding
        and will output an encoded representation
        of shape (B, T, C).

        First, we need to create an embedding from the sequence
        of tokens. For this, we need the vocab size.

        Next, we want to create a series of Decoder layers.
        For this, we need to specify the number of layers
        and the number of heads.

        Additionally, for every Multi-Head Attention layer, we
        need to know how long each query/key is, and how long
        each value is.
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.ffn_hidden_dim = ffn_hidden_dim

        self.qk_length = qk_length
        self.value_length = value_length

        # Define any layers you'll need in the forward pass
        # Hint: You may find `ModuleList`s useful for creating
        # multiple layers in some kind of list comprehension.
        #
        # Recall that the input is just a sequence of tokens,
        # so we'll have to first create some kind of embedding
        # and then use the other layers we've implemented to
        # build out the Transformer decoder.

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = PositionalEncoding(d_model=embedding_dim, dropout=dropout, max_len=max_length)
        self.output_projection = nn.Linear(embedding_dim, vocab_size)

        self.layers = nn.ModuleList([
            DecoderLayer(
                num_heads=num_heads,
                embedding_dim=embedding_dim,
                ffn_hidden_dim=ffn_hidden_dim,
                qk_length=qk_length,
                value_length=value_length,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        # raise NotImplementedError("Need to implement Decoder layers")

    def forward(
        self,
        x: torch.Tensor,
        enc_x: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        The forward pass of the Decoder.
        """
        x = x.long()

        x = self.token_embedding(x)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, enc_x=enc_x, tgt_mask=tgt_mask, src_mask=src_mask)
        return self.output_projection(x)
        raise NotImplementedError("Need to implement forward pass of Decoder")
