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
        self.cosine_sim_logits = config.cosine_sim_logits
        self.radius = config.radius if config.radius is not None else 1
        self.word_embeddings = nn.Parameter(torch.Tensor(config.vocab_size, config.embedding_dims))
        self.position_embeddings = None
        self.pos_padding_idx = config.max_length
        self.decoder = TransformerEncoder(config=config)
        # self.project_hidden_states = nn.Linear(in_features=config.embedding_dims, out_features=config.embedding_dims, bias=False)
        self.w_out = None
        if not config.tied_embeddings:
            self.w_out = nn.Parameter(torch.Tensor(config.vocab_size, config.embedding_dims))
            nn.init.normal_(self.w_out, mean=0, std=sqrt(2 / (2 * self.embedding_dims)))
        if config.add_positions:
            self.position_embeddings = nn.Parameter(torch.Tensor(config.max_length + 1, config.embedding_dims))
        self.word_padding_idx: int = config.pad_idx
        self.embedding_dims = config.embedding_dims
        # initializing word embeddings
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
        b, s = x.shape
        device = x.device
        # todo: no tied embedding when using similarity logits
        # also, is'nt weird to normalize the embedding at the net input ?
        w_embeddings = self.word_embeddings
        e = F.embedding(input=x, weight=w_embeddings, padding_idx=self.word_padding_idx)

        if self.position_embeddings is not None:
            positions = torch.arange(0, s).unsqueeze(0).repeat(b, 1).to(device)
            pad_positions = (x == self.word_padding_idx).to(device)
            positions[pad_positions] = self.pos_padding_idx
            p_embeddings = F.embedding(input=positions, weight=self.position_embeddings, padding_idx=self.pos_padding_idx)
            e += p_embeddings
        mask = torch.ones(s, s).triu(1).bool().to(device)
        c = self.decoder(e, mask)
        # c = self.project_hidden_states(c)
        if self.cosine_sim_logits:
            # fix the norm of the hidden and embedding vectors to self.radius
            w_embeddings = self.radius * F.normalize(w_embeddings, dim=-1)
            c = self.radius * F.normalize(c, dim=-1)
        if self.w_out is None:
            # tied embeddings
            return c @ w_embeddings.T
        return c @ self.w_out.T