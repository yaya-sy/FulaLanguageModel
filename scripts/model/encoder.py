"""This module implements a transformer encoder."""
# native python packages imports
import sys
sys.path.append('.')
from typing import Optional
from scripts.config.config import Config
from scripts.model.transformer_layer import TransformerLayer

# installed python packages imports
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
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.layers)])

    def forward(self,
                src: Tensor,
                mask: Optional[Tensor]=None
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

        for transformer in self.layers:
            src = transformer(src, src, mask=mask)
        return src