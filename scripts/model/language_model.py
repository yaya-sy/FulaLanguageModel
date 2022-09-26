from encoder import TransformerEncoder

from torch import nn
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
        self.decoder = TransformerEncoder(config=config)
        self.linear = nn.Linear(config.embedding_dims, config.vocab_size)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward across all the transformer decoder layers.

        Parameters
        ----------
        - x: Tensor
            The input sequences. Must be of shape [b, s] where 'b'\
            is the batch size, 's' the length of the sequences in the batch.
        """

        c = self.decoder(x)
        return self.linear(c)