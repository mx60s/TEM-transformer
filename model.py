#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a pytorch implementation of TEM-t (or TEM-transformer)
based on the paper here: https://arxiv.org/abs/2112.04035 and Jacob Bakerman's TEM Pytorch implementation here: https://github.com/jbakermans/torch_tem

I'm not sure if I'll actually finish this

@author: mx60s
"""
import copy
import math
import os
import pdb
from tempfile import TemporaryDirectory
from typing import Tuple

# Standard modules
import numpy as np
import torch
from scipy.stats import truncnorm
from torch import Tensor, nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

# Custom modules
import utils


class Model(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.hyper = copy.deepcopy(params)
        self.init_trainable()

    def forward(self, x):
        raise NotImplementedError

    def iteration(self, x):
        raise NotImplementedError

    def inference(self, x):
        raise NotImplementedError

    def generative(self):
        raise NotImplementedError

    def loss(self):
        raise NotImplementedError

    def init_trainable(self):
        raise NotImplementedError


# Might need to rename since these are essentially g
class PositionalEncoding(nn.Module):
    def __init__(self, params):
        super().__init__()
        dropout = params["dropout"]
        max_len = params["max_len"]
        d_model = params["d_model"]

        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Linear(
            max_len, d_model
        )  # dim was (max_len, 1, d_model) in original, may need to reshape
        # I think this also needs nonlinearity but it doesn't say what in the paper
        # look for other examples in the original TEM code of these learnable matrices
        # g_t+1 = f(g_t * W_a) where W_a is learnable action-dependent matrix

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x += self.pe
        return self.dropout(x)  # do we still want a dropout?


class TransformerEmbedding(nn.Module):
    def __init__(self):
        # combine a positional encoding w an embedding
        # paper mentions some embedding but not sure
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor


# layernorm on the positional encodings (not in the RNN, but on the input to transformer)
# we use fixed weights on the layer norm, i.e. is is just a z score of g


# TODO explore using learnable layer norm
class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)  # TODO replace w z-score
        self.dropout1 = nn.Dropout(p=drop_prob)

        # TODO this is dumb replace
        # self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)  # TODO replace w z-score
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        # 1. compute self attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)

        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        enc_voc_size,
        max_len,
        d_model,
        ffn_hidden,
        n_head,
        n_layers,
        drop_prob,
        device,
    ):
        super().__init__()
        self.emb = TransformerEmbedding()

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model,
                    ffn_hidden=ffn_hidden,
                    n_head=n_head,
                    drop_prob=drop_prob,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, src_mask):
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x


class Iteration:
    def __init__(
        self,
        g=None,
        x=None,
        a=None,
        L=None,
        M=None,
        g_gen=None,
        p_gen=None,
        x_gen=None,
        x_logits=None,
        x_inf=None,
        g_inf=None,
        p_inf=None,
    ):
        # Copy all inputs
        self.g = g
        self.x = x
        self.a = a
        self.L = L
        self.M = M
        self.g_gen = g_gen
        self.p_gen = p_gen
        self.x_gen = x_gen
        self.x_logits = x_logits
        self.x_inf = x_inf
        self.g_inf = g_inf
        self.p_inf = p_inf

    def correct(self):
        # Detach observation and all predictions
        observation = self.x.detach().numpy()
        predictions = [tensor.detach().numpy() for tensor in self.x_gen]
        # Did the model predict the right observation in this iteration?
        accuracy = [
            np.argmax(prediction, axis=-1) == np.argmax(observation, axis=-1)
            for prediction in predictions
        ]
        return accuracy

    def detach(self):
        # Detach all tensors contained in this iteration
        self.L = [tensor.detach() for tensor in self.L]
        self.M = [tensor.detach() for tensor in self.M]
        self.g_gen = [tensor.detach() for tensor in self.g_gen]
        self.p_gen = [tensor.detach() for tensor in self.p_gen]
        self.x_gen = [tensor.detach() for tensor in self.x_gen]
        self.x_inf = [tensor.detach() for tensor in self.x_inf]
        self.g_inf = [tensor.detach() for tensor in self.g_inf]
        self.p_inf = [tensor.detach() for tensor in self.p_inf]
        # Return self after detaching everything
        return self
