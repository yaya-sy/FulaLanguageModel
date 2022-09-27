"""Implementation of one transformer layer."""
# standard python imports
import sys
sys.path.append('.')
from typing import Optional
from scripts.config.config import Config

# non standard python libraries imports
import torch
from torch import nn
from torch import Tensor

class MultiHeadAttention(nn.Module):
    """
    This class implements a multihead attention.

    Parameters
    ----------
    - config: Config
        An instance of the class Config containing all the hyperparameters
        and options.
    """
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.K = nn.Linear(in_features=config.embedding_dims, out_features=config.embedding_dims)
        self.Q = nn.Linear(in_features=config.embedding_dims, out_features=config.embedding_dims)
        self.V = nn.Linear(in_features=config.embedding_dims, out_features=config.embedding_dims)
        self.heads = config.heads
        self.softmax = nn.Softmax(-1)
        self.dropout_mha = nn.Dropout(p=config.dropout)
    
    def forward(self,
                q: Tensor,
                k: Tensor,
                v: Tensor,
                keys_pad_mask: Optional[Tensor]=None,
                mask_futur: bool=True
                ) -> Tensor:
        """
        Compute attention mechanism of an embedded sequence of tokens.

        Parameters
        ----------
        - q: Tensor
            The queries as a three dimension tensor of embedded sequences.
            The shape must be [b, s, e] where 'b' is the batch size,\
            's' is the lengths of the sequences and 'e' the embedding dimension.
            
        - k: Tensor
            The keys as a three dimension tensor of embedded sequences.
            The shape must be [b, s, e] where 'b' is the batch size,\
            's' is the lengths of the sequences and 'e' the embedding dimension.

        - v: Tensor
            The values as a three dimension tensor of embedded sequences.
            The shape must be [b, s, e] where 'b' is the batch size,\
            's' is the length of the sequence and 'e' the embedding dimension.

        - keys_pad_mask: Tensor or None.
            Do not mask to the subsequent keys indexes.
            If given, must be a boolean tensor of shape [b, s_k].
            The boolean value 'True' will be masked.
        
        - mask_futur: bool
            If True, the next tokens of the keys will be masked
            by masking the lower triangle of the matrix.
        
        Returns
        -------
        - Tensor:
            Tensor of shape [b, s_q, c] where 'b' is the batch size,\
            's_q' is the length of the query sequence and 'c' the reduction of\
            the concatenation of the output of each attention model.
            'c' is reduced since the keys, queries and values projections\
            of the tokens of the sequence are divided by the number of heads.

            The returned vector represents for each tokens the concatenation\
            of its attention vectors coming from all the heads.
        """
  
        b, s_q, _ = q.shape # [batch_size, seq_length, embedd_dims]
        _, s_k, _ = k.shape 
        _, s_v, _ = v.shape

        # linear projections of the keys, queries and values.
        # and split them into the number of heads, before to compute the attention
        # so we have multiple attention models (heads) learned independantly.
        K = self.K(k).view(b, self.heads, s_k, -1)
        Q = self.Q(q).view(b, self.heads, s_q, -1)
        V = self.V(v).view(b, self.heads, s_v, -1)


        # dot-product and use the scaling factor from 'Attention is all you need" paper
        # (https://arxiv.org/pdf/1706.03762.pdf)
        QK = (Q @ K.transpose(2, 3)) / (1 / torch.sqrt(torch.tensor(Q.shape[-1]))) # shape=[b, h, s_q, s_k]
        
        # mask the pad indexes
        if keys_pad_mask is not None:
            pad_mask = keys_pad_mask.view(b, 1, 1, -1).repeat(1, self.heads, s_q, 1)
            QK[pad_mask] = float("-inf")
        # mask futur, relevant for a decoder language model for example
        if mask_futur:
            attn_mask = torch.triu(torch.ones(1, 1, s_q, s_k), 1).repeat(b, self.heads, 1, 1).bool()
            QK[attn_mask] = float("-inf")
        attention = self.softmax(QK) # shape=[b, h, s, s])
        # for each word, concatenate the attention vectors comming from all the heads.
        out = (attention @ V).view(b, s_q, -1) # [b, s, embedd_dims]
        return self.dropout_mha(out)

class TransformerLayer(nn.Module):
    """
    This class implements one tranfromer block.

    Parameters
    ----------
    - config: Config
        An instance of the class Config containing all the hyperparameters
        and options.
    """
    def __init__(self, config):
        super().__init__()
        self.mha = MultiHeadAttention(config)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=config.embedding_dims)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=config.embedding_dims, out_features=config.ff_size),
            nn.GELU(),
            nn.Linear(in_features=config.ff_size, out_features=config.embedding_dims)
        )

        self.dropout_ff = nn.Dropout(p=config.dropout)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=config.embedding_dims)
    
    def forward(self,
                src: Tensor,
                tgt: Tensor,
                keys_pad_mask: Optional[Tensor]=None,
                mask_futur: bool=False
                ) -> Tensor:
        """
        Forward an embedded tensor in the transformer layer.

        Parameters
        ----------
        - src: Tensor
            The three dimension tensor of the embedded sequence.
            The shape must be [b, s, e] where 'b' is the batch size,\
            's' is the length of the source sequence and 'e' the embedding dimension.
        - tgt: Tensor
            The three dimension tensor of the embedded sequence.
            The shape must be [b, s, e] where 'b' is the batch size,\
            's' is the length of the target sequence and 'e' the embedding dimension.

        - keys_pad_mask: Tensor or None.
            Mask the pad indexes. If given, must be a boolean tensor of shape [b, s].
            The boolean value 'True' will be masked.
        
        - mask_futur: bool
            If True, the next tokens of the keys will be masked\
            using by masking the upper triangle of the matrix.
        
        Returns
        -------
        - Tensor:
            Tensor of shape [b, s, c] where 'b' is the batch size,\
            's' is the length of the sequence and 'c' the contextual\
            embedding dimension.
            This represents the contextual vector for each vector in the\
            the sequence. So each vector in the sequence contains\
            the informations about other tokens in the sequence.
        """
        c = self.mha(src, tgt, tgt, keys_pad_mask, mask_futur)
        cz = self.layer_norm1(c + tgt)
        f = self.dropout_ff(self.mlp(cz))
        return self.layer_norm2(f + cz)