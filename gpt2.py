import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import init_weights

class GPT2Model(nn.Module):
    """
    GPT-2 model implementation.
    """
    def __init__(self, vocab_size, n_ctx, n_embd, n_head, n_layer,
                 activation_fn='gelu', pad_idx=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.activation_fn = activation_fn
        self.pad_idx = pad_idx

        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(n_ctx, n_embd)
        self.drop = nn.Dropout(0.1)
        self.h = nn.ModuleList([
            Block(n_ctx, n_embd, n_head, n_layer, activation_fn)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)

        self.apply(init_weights)

    def forward(self, x):
        # x: (batch_size, seq_len)
        # mask: (batch_size, seq_len)
        batch_size, seq_len = x.size()
        mask = (x != self.pad_idx).unsqueeze(-2).expand(batch_size, seq_len, seq_len)

        h = self.wte(x)
        h = h + self.wpe(torch.arange(seq_len, device=x.device))
        h = self.drop(h)

        for block in self.h:
            h = block(h, mask)

        return self.ln_f(h)

class Block(nn.Module):
    """
    GPT-2 block implementation.
    """
    def __init__(self, n_ctx, n_embd, n_head, n_layer, activation_fn):
        super().__init__()
        self.attn = Attention(n_ctx, n_embd, n_head)
        self.ln_1 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, activation_fn)
        self.ln_2 = nn.LayerNorm(n_embd)

    def forward(self, x, mask):
        # x: (batch_size, seq_len, n_embd)
        # mask: (batch_size, seq_len, seq_len)
        h = self.attn(x, mask)
        h = self.ln_1(x + h)
        h = self.mlp(h)
        return self.ln_2(h + x)

class Attention(nn.Module):
    """
    GPT-2 attention implementation.
    """
    def __init__(self, n_ctx, n_embd, n_head):
        super().__init__()
        self.c_attn = nn.Linear(n_embd, n_head * n_embd)
        self.c_proj = nn.Linear(n_embd, n_head * n_embd)
        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)

    def forward(self, x, mask):
        # x: (batch_size, seq_len, n_embd)
        # mask: (batch_size, seq_len, seq_len)
        batch_size, seq_len, n_embd = x.size()
        h_attn = self.c_attn(x).view(batch_size, seq_len, n_head, n_embd)
        h_proj = self.c_proj(x).view(batch_size, seq_len, n_head, n_embd)

        # (batch_size, n_head, seq_len, seq_len)
        attn_score = torch.einsum('bhld,bhmd->bhml', (h_attn, h_proj))
        attn_score = attn_score.masked_fill(mask == 0, -1e9)
        attn_prob = F.softmax(attn_score, dim=-1)
        attn_prob = self.attn_dropout(attn_prob)

        # (batch_size, n_head, seq_len, n_embd)
        attn_vec = torch.einsum('bhml,bhmd->bhld', (attn_prob, h_proj))
        attn_vec = attn_vec.contiguous().view(batch_size, seq_len, n_head * n_embd)

        attn_out = self.resid_dropout(attn_vec)
        return attn_out

class MLP(nn.Module):
    """
    GPT-2 MLP implementation.
    """
    def __init__(self, n_embd, activation_fn):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, n_embd * 4)
        self.c_proj = nn.Linear(n_embd * 4, n_embd)
        self.activation_fn = activation_fn
        self.mlp_dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: (batch_size, seq_len, n_embd)
        h = self.c_fc(x)
        h = self.activation_fn(h)
        h = self.mlp_dropout(h)
        h = self.c_proj(h)
        return h