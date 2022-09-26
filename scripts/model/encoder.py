# native python packages imports
from typing import Union
from .config.config import Config
from transformer_layer import TransformerLayer

# installed python packages imports
import torch
from torch import nn
from torch import Tensor

class TransformerEncoder(nn.Module):
    """
    This class implements an attention based\
    sequences encoder using the transformer architecture.

    Parameters
    ----------
    - config: Config
        An instance of the class Config, containing\
        all the relevant parameters.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.transformers = nn.ModuleDict(
            {
                "w_embeddings" : nn.Embedding(num_embeddings=config.vocab_size,
                                               embedding_dim=config.embedding_dims,
                                               padding_idx=config.pad_idx),
                "layers" : nn.ModuleList([TransformerLayer(config) for _ in range(config.layers)])
            })
        if config.add_positions:
            self.transformers["p_embeddings"] = nn.Embedding(num_embeddings=config.max_length,
                                                              embedding_dim=config.embedding_dims,
                                                              padding_idx=config.pad_idx)
        self.pad_idx: int = config.pad_idx

    def forward(self,
                src: Tensor,
                mask_pad_idxs: bool=True,
                mask_futur: bool=False
                ) -> Tensor:
        """
        Will embedded the input sequences and represent them\
        contextually across all the transformer layers.

        Parameters
        ----------
        - src: Tensor
            A batch of a unambedded sequence where each token is represented\
            by an index.
            The shape must be [b, s] where 'b' is the batch size,
            's' is the length of the source sequence.

        - pad_mask: bool
            Whether or not mask the padding idexes.
        
        - mask_futur: bool
            If True, the next tokens will be masked using by masking the upper
            triangle of the matrix.
        
        Returns
        -------
        - Tensor:
            Tensor of shape [b, s, c] where 'b' is the batch size,
            's' is the length of the sequence and 'c' the contextual\
            embedding dimension. The output vectors represent the contextual\
            representation of the sequences across all the transformer layers.
        """
        _, s = src.shape
        device: str = src.device

        pad_mask_true: bool = mask_pad_idxs and self.pad_idx is not None
        pad_mask: Union[Tensor, None] = (src == self.pad_idx).to(device) if pad_mask_true else None

        e = self.transformers.w_embeddings(src)

        if "p_embeddings" in self.transformers:
            positions = torch.arange(0, s, dtype=torch.long, device=device).unsqueeze(0)
            p_embeddings = self.transformers.p_embeddings(positions)
            e += p_embeddings

        for transformer in self.transformers.layers:
            e = transformer(e, e, keys_pad_mask=pad_mask, mask_futur=mask_futur)
        return e