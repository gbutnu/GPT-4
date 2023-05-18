"""Position embeddings for GPT-4."""

import torch
import torch.nn as nn

class PositionEmbeddings(nn.Module):
    """Position embeddings for GPT-4.

    This module implements position embeddings for GPT-4. It takes in a
    sequence of token ids and outputs a sequence of position embeddings.

    Args:
        num_positions (int): The maximum number of positions to embed.
        embedding_dim (int): The dimension of the position embeddings.
    """

    def __init__(self, num_positions, embedding_dim):
        super().__init__()
        self.num_positions = num_positions
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(num_positions, embedding_dim)

    def forward(self, inputs):
        """Forward pass of the position embeddings.

        Args:
            inputs (torch.Tensor): A tensor of shape (batch_size, sequence_length)
                containing the token ids.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, sequence_length, embedding_dim)
                containing the position embeddings.
        """
        positions = torch.arange(inputs.size(1), device=inputs.device).expand(inputs.size())
        positions = positions.long()
        positions = positions.unsqueeze(0).expand_as(inputs)
        positions = positions.clamp(max=self.num_positions - 1)
        return self.embeddings(positions)
