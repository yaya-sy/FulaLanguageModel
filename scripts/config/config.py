"""
This module implements a class for storing\
all relevant parameters and/or hyperparameters\
for the language model.
"""
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
    - epochs: int
        The number of epochs.
    - tokenizer: str
        Path to the trained sentencepiece tokenizer
    - train: str
        Path to the training corpus.
    - dev: str
        Path to the development corpus.
    - batch_size: int
        The batch size.
    - lr: float, int
        The learning rate.
    - print_generation_steps: int
        Print the generation every given steps (batchs)
    - checkpoint: str
        If given, the path to a given checkpoint of the model
    - T_max: int
        Decreases the lr to for many iterations.
    - gradients_accumulation: int
        The number of examples to accumulate
    """
    vocab_size: int
    tokenizer: str
    train: str
    dev: str
    epochs: int
    batch_size: int
    lr: Union[int, float]
    pad_idx: Optional[int]
    embedding_dims: int
    max_length: int
    add_positions: bool
    heads: int
    layers: int
    ff_size: int
    dropout: Union[int, float]
    valid_every_n_batchs: int
    T_max: int
    checkpoint: Optional[str]
    gradients_accumulation: int

    def to_dict(self) -> Dict[str, Number]:
        """Will return all the parameters as a dictionnary."""
        return asdict(self)