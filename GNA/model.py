import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerModel(nn.Module):
    def __init__(self, vocab_size=2, d_model=128, n_head=8, num_layers=6, max_seq_len=100):
        """
        Parameters:
        - vocab_size: Number of discrete tokens (for binary variables, this is 2).
        - d_model: The embedding dimension.
        - n_head: The number of attention heads.
        - num_layers: Number of transformer layers.
        - max_seq_len: Maximum length of the input sequence (number of binary variables).
        """

        super(TransformerModel, self).__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model, n_head, dim_feedforward=d_model, dropout=0), num_layers)
        self.linear = nn.Linear(d_model, vocab_size)
        self.beta_projection = nn.Linear(1, d_model)

    @staticmethod
    def _generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x, beta):
        seq_len, batch_size = x.size()
        device = x.device

        # Get token embeddings and add positional embeddings
        token_emb = self.token_embedding(x.int())  # (seq_len, batch_size, d_model)
        positions = torch.arange(seq_len, device=device).unsqueeze(1).expand(seq_len, batch_size)
        pos_emb = self.position_embedding(positions)  # (seq_len, batch_size, d_model)
        if torch.is_tensor(beta):
            beta_emb = self.beta_projection(beta.log().unsqueeze(1))  # (batch_size, d_model)
        else:
            beta_emb = self.beta_projection(torch.tensor(beta, dtype=torch.get_default_dtype()).log().reshape(1, 1)
                                            .expand(batch_size, -1))
        x = token_emb + pos_emb + beta_emb  # (seq_len, batch_size, d_model)
        # x = token_emb + pos_emb

        mask = self._generate_square_subsequent_mask(len(x)).to(x.device)
        x = self.transformer_encoder(x, mask)

        # Map to logits over the binary tokens (0 and 1)
        logits = self.linear(x)  # (seq_len, batch_size, vocab_size)

        return logits
