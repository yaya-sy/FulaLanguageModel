# standard python imports
from typing import Union, Optional, Dict
from dataclasses import dataclass, asdict
from numbers import Number

@dataclass
class Config:
    """
    A config class for the hyperparameters\
    of the language model.
    
    Parameters
    ----------
    - vocab_size: int
        The number of unique tokens of the language model.
    - pad_index: int, None
        The index of the pad token.
    - embedding_dims: int
        The size of the vector representing the tokens
    - max_length: int
        The longest sequence 
    - add_positions: bool
        Whether or not encode the positions of the sequences with\
        embeddings.
    - heads: int
        The number of heads for the multihead attention mechanism.
    - layers: int
        The number of layer of the transformer encoder or decoder.
    - ff_size: int
        The size of the hidden layers for the MLP layer in each\
        transformer layer.
    - dropout: int, float
        The percentage of the neurons to be droped.
    """
    vocab_size: int
    pad_idx: Optional[int] = 3
    embedding_dims: int = 512
    max_length: int = 200
    add_positions: bool = True
    heads: int = 8
    layers: int = 12
    ff_size: int = heads * embedding_dims
    dropout: Union[int, float] = 0.07

    def to_dict(self) -> Dict[str, Number]:
        """Will return all the parameters as a dictionnary."""
        return asdict(self)
    
    def init_from_dict(self,
                       dictionnary: Dict[str, Number]):
        """
        Will return a new instance of the class\
        from a given dictionnary
        """
        return Config(**dictionnary)