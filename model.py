import torch
import torch.nn as nn

class GPT4Model(nn.Module):
    """
    GPT-4 Model
    """
    def __init__(self, config):
        super(GPT4Model, self).__init__()
        self.config = config
        self.transformer = nn.Transformer(
            d_model=config.d_model,
            nhead=config.nhead,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation
        )
        self.output_layer = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, x, mask):
        output = self.transformer(x, mask)
        output = self.output_layer(output)
        return output