"""TODO"""
from typing import List, Tuple, Iterator
from ..config.config import Config

Sequence = List[str]
EncodedSequence: List[int]
Batch = List[EncodedSequence]
Example = Tuple[EncodedSequence, EncodedSequence]

class BatchGenerator:
    """TODO"""

    def __init__(self,
                 bpe_model,
                 config: Config,
                 text):
        self.bpe_model = bpe_model
        self.config = config
        self.examples = self.make_examples(text)
        pass

    def pad(self, batch: Batch) -> Batch:
        """TODO"""
        batch
        pass

    def encode(self,
               sequence: Sequence
               ) -> EncodedSequence:
        """TODO"""
        sequence
        pass

    def decode(self,
               encoded_sequence: EncodedSequence
               ) -> Sequence:
        """TODO"""
        encoded_sequence
        pass

    def example(self,
                sequence: Sequence
                ) -> Example:
        """TODO"""
        sequence
        pass
    
    def make_examples(self, text) -> List[Example]:
        """TODO"""
        pass

    def prompt(self) -> EncodedSequence:
        """TODO"""
        pass

    def __getitem__(self, idx) -> Example:
        """TODO"""
        return self.examples[idx]

    def __call__(self,
                 batch_size: int=32
                 ) -> Iterator[Batch]:
        """TODO"""
        batch_size
        pass
