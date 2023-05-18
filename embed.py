import torch
import torch.nn as nn

class Embedding(nn.Module):
    """
    A simple embedding layer.
    """
    def __init__(self, vocab_size, embedding_dim):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x)