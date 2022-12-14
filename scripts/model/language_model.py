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

class FixNorm(nn.Module):
    """FixNorm"""
    def __init__(self, radius):
        super(FixNorm, self).__init__()
        self.radius = nn.Parameter(torch.ones(1) * radius)

    def forward(self, x):
        return self.radius * F.normalize(x, dim=-1)

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
        self.radius = torch.tensor(float(config.radius))
        self.word_embeddings = nn.Parameter(torch.Tensor(config.vocab_size, config.embedding_dims))
        self.position_embeddings = None
        self.pos_padding_idx = config.max_length
        self.decoder = TransformerEncoder(config=config)
        self.linear = None
        if not config.tied_embeddings:
            self.linear = nn.Linear(config.embedding_dims, config.vocab_size)
        else:
            # bias is a vector 
            self.out_bias = nn.Parameter(torch.Tensor(config.vocab_size))
            torch.nn.init.zeros_(self.out_bias)
        if config.add_positions:
            self.position_embeddings = nn.Parameter(torch.Tensor(config.max_length + 1, config.embedding_dims))
        self.word_padding_idx: int = config.pad_idx
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
    
    def forward(self, x: Tensor, apply_mask=True) -> Tensor:
        """
        Forward across all the transformer decoder layers.

        Parameters
        ----------
        - x: Tensor
            The input sequences. Must be of shape [b, s] where 'b'\
            is the batch size, 's' the length of the sequences in the batch.
        """
        b, s = x.shape
        device = x.device
        w_embeddings = self.radius * F.normalize(self.word_embeddings, dim=-1) if self.fix_norm else self.word_embeddings
        e = F.embedding(input=x, weight=w_embeddings, padding_idx=self.word_padding_idx)

        if self.position_embeddings is not None:
            positions = torch.arange(0, s).unsqueeze(0).repeat(b, 1).to(device)
            pad_positions = (x == self.word_padding_idx).to(device)
            positions[pad_positions] = self.pos_padding_idx
            p_embeddings = F.embedding(input=positions, weight=self.position_embeddings, padding_idx=self.pos_padding_idx)
            e += p_embeddings
        mask = torch.ones(s, s).triu(1).bool().to(device)
        c = self.decoder(e, mask)
        if self.fix_norm:
            # fix the norm of the hidden (or contextual) vectors to self.radius too
            c = self.radius * F.normalize(c, dim=-1)
        if self.linear is None:
            # tied embeddings
            return F.linear(input=c, weight=w_embeddings, bias=self.out_bias)
        return self.linear(c)