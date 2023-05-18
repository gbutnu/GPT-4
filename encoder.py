import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.enc_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    
    Args:
        config: an instance of :class:`~transformers.PretrainConfig`
        embed_tokens (torch.nn.Embedding): input embeddings
    """
    def __init__(self, config, embed_tokens):
        super().__init__()
        self.dropout = config.dropout
        self.layerdrop = config.layerdrop

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(config.d_model)
        self.embed_positions = PositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            self.dropout,
            self.layerdrop,
        )

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(config)
            for i in range(config.enc_layers)
        ])
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = config.normalize_before
        if self.normalize:
            self.layer_norm = LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(self, src_tokens, src_lengths):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(src_tokens)
        x += self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        if self.normalize:
            x = self.layer_norm(x)

        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }