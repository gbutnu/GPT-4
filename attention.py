import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    Attention mechanism used in the GPT-4 model.
    """
    def __init__(self, n_heads, n_units, dropout=0.1):
        """
        Initialize the Attention mechanism.
        :param n_heads: Number of attention heads.
        :param n_units: Number of units in the attention layer.
        :param dropout: Dropout rate.
        """
        super(Attention, self).__init__()
        self.n_heads = n_heads
        self.n_units = n_units
        self.dropout = dropout

        # Linear layers for the three vectors used to compute the attention weights.
        self.q_linear = nn.Linear(n_units, n_units)
        self.k_linear = nn.Linear(n_units, n_units)
        self.v_linear = nn.Linear(n_units, n_units)

        # Output linear layer.
        self.out = nn.Linear(n_units, n_units)

        # Dropout layer.
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        Compute the attention weights.
        :param query: Query vector.
        :param key: Key vector.
        :param value: Value vector.
        :param mask: Mask to prevent attention to certain values (optional).
        :return: The attended vector.
        """
        # Compute the query, key and value vectors.
        q = self.q_linear(query)
        k = self.k_linear(key)
        v = self.v_linear(value)

        # Compute the attention weights.
        weights = torch.matmul(q, k.transpose(1, 2))
        weights = weights / (self.n_units ** 0.5)

        # Apply the mask (if provided).
        if mask is not None:
            weights = weights.masked_fill(mask == 0, -1e9)

        # Normalize the weights.
        weights = F.softmax(weights, dim=-1)

        # Apply the dropout.
        weights = self.dropout(weights)

        # Compute the attended vector.
        attended = torch.matmul(weights, v)

        # Compute the output vector.
        output = self.out(attended)

        return output
