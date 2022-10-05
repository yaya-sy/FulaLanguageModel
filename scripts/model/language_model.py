"""
Implementation of a Transformer language model\
by using the transformer encoder with masked next token.
"""
import sys
sys.path.append('.')
from scripts.model.encoder import TransformerEncoder

import torch
from torch import nn
from torch.amp import autocast
from torch import Tensor

class TransformerLM(nn.Module):
    """
    This class implements a transformer based language model.
    The model uses a teansformer encoder where the next tokens\
    are masked.

    Parameters
    ----------
    - config: Config
        An instance of the class Config, containing\
        all the relevant parameters.
    """

    def __init__(self, config):
        # decoder = encoder but with masked next tokens
        super().__init__()
        self.transformer_lm = nn.ModuleDict(
            {
                "w_embeddings" : nn.Embedding(num_embeddings=config.vocab_size,
                                               embedding_dim=config.embedding_dims,
                                               padding_idx=config.pad_idx),
                "decoder" : TransformerEncoder(config=config),
                "linear" : nn.Linear(config.embedding_dims, config.vocab_size),
            })
        if config.add_positions:
            self.transformer_lm["p_embeddings"] = nn.Embedding(num_embeddings=config.max_length,
                                                              embedding_dim=config.embedding_dims,
                                                              padding_idx=config.pad_idx)
        self.pad_idx: int = config.pad_idx
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward across all the transformer decoder layers.

        Parameters
        ----------
        - x: Tensor
            The input sequences. Must be of shape [b, s] where 'b'\
            is the batch size, 's' the length of the sequences in the batch.
        """
        _, s = x.shape
        device = x.device

        # masking future
        mask = torch.empty(s, s).to(device)
        mask.fill_(float("-inf")).triu_(1)
        # masking pad indexes
        pad_mask = (x == self.pad_idx)[-1, ...]
        mask[:, pad_mask] = float("-inf")

        e = self.transformer_lm.w_embeddings(x)

        if "p_embeddings" in self.transformer_lm:
            positions = torch.arange(0, s, dtype=torch.long, device=device).unsqueeze(0)
            p_embeddings = self.transformer_lm.p_embeddings(positions)
            e += p_embeddings

        c = self.transformer_lm.decoder(e, mask)
        return self.transformer_lm.linear(c)