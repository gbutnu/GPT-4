"""
Transformer models for OpenAI GPT-4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from openai.gpt4.modeling_gpt4 import GPT4PreTrainedModel, GPT4Model


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer Blocks for use in GPT-4
    """

    def __init__(self, hidden_size, num_attention_heads, intermediate_size=None,
                 attention_probs_dropout_prob=0.1,
                 hidden_dropout_prob=0.1,
                 activation="gelu",
                 layer_norm_eps=1e-12):
        super(TransformerBlock, self).__init__()
        self.attention = GPT4Model.GPT4Attention(
            hidden_size, num_attention_heads,
            attention_probs_dropout_prob,
            intermediate_size=intermediate_size,
            activation=activation,
            layer_norm_eps=layer_norm_eps)
        self.intermediate = GPT4Model.GPT4Intermediate(
            hidden_size, intermediate_size,
            activation=activation,
            layer_norm_eps=layer_norm_eps)
        self.output = GPT4Model.GPT4Output(
            hidden_size, hidden_dropout_prob,
            layer_norm_eps=layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        intermediate_output = self.intermediate(attention_outputs[0])
        layer_output = self.output(intermediate_output, attention_outputs[1])
        return layer_output


class GPT4Transformer(GPT4PreTrainedModel):
    """OpenAI GPT-4 Transformer with a language modeling head."""

    def __init__(self, config):
        super(GPT4Transformer, self).__init__(config)
        self.num_labels = config.num_labels

        self.transformer = GPT4Model(config)
        self.lm_head = GPT4Model.GPT4LMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def forward(self, input_ids=None, past=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):

        transformer_outputs = self.transformer(input_ids,
                                              past=past,
                                              attention_mask=attention_mask,
                                              token_type_ids=token_type_ids,
                                              position_ids=position_ids,
                                              head_mask=head_mask)

        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)