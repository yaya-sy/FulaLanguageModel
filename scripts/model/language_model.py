"""
Implementation of a Transformer language model\
by using the transformer encoder with masked next token.
"""
import sys
sys.path.append('.')
from scripts.model.encoder import TransformerEncoder
from math import sqrt

import torch
from torch import nn
import torch.nn.functional as F
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
        self.fix_norm = config.normalize_word_embeddings
        self.r = config.radius
        self.word_embeddings = nn.Parameter(torch.Tensor(config.vocab_size, config.embedding_dims))
        self.position_embeddings = None
        self.decoder = TransformerEncoder(config=config)
        self.linear = None
        if not config.tied_embeddings:
            self.linear = nn.Linear(config.embedding_dims, config.vocab_size)
        else:
            # bias is a vector 
            self.out_bias = nn.Parameter(torch.Tensor(config.vocab_size))
        if config.add_positions:
            self.position_embeddings = nn.Parameter(torch.Tensor(config.max_length, config.embedding_dims))
        self.pad_idx: int = config.pad_idx
        self.embedding_dims = config.embedding_dims
        # initializing word embeddings
        if self.fix_norm:
            d = 0.01
            nn.init.uniform_(self.word_embeddings, a=-d, b=d)
        else:
            # xavier normal
            nn.init.normal_(self.word_embeddings, mean=0, std=sqrt(2 / (2 * self.embedding_dims)))
        # initializing the rest
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        std = sqrt(2 / (2 * self.embedding_dims))
        if isinstance(module, nn.Linear):
            # xavier normal
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
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
        mask = torch.zeros(s, s).bool().to(device)
        # masking pad indexes
        pad_mask = (x == self.pad_idx)[-1, ...]
        mask[:, pad_mask] = True

        w_embeddings = (self.r * F.normalize(self.word_embeddings, dim=-1)
                        if self.fix_norm else self.word_embeddings)
        e = F.embedding(input=x, weight=w_embeddings, padding_idx=self.pad_idx)

        if self.position_embeddings is not None:
            positions = torch.arange(0, s, dtype=torch.long, device=device).unsqueeze(0)
            p_embeddings = F.embedding(input=positions, weight=self.position_embeddings, padding_idx=self.pad_idx)
            e += p_embeddings

        c = self.decoder(e, mask)
        if self.linear is None:
            # tied embeddings
            return (c @ w_embeddings.T) + self.out_bias
        return self.linear(c)