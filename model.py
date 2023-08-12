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

    def forward(self, walk: Walk, prev_iter=None):
        # given an observation, and knowing the previous g (or starting from the beginning)
        # construct a p from the outer product of these x and g
        # feed into transformer and get a p for the next time step which is broken down into x and g
        # g_curr = new g for the next iteration
        # compare our x with ground truth observation x for loss
        # is constructed/deconstructed outside of transformer though I think

        steps = self.init_walks(prev_iter)
        for g, x, a in walk:
            if steps is None:
                steps = [
                    self.init_iteration(g, x, [None for _ in range(len(a))])
                ]
            L, M, g_gen, p_gen, x_gen, x_logits, x_inf, g_inf, p_inf = self.iteration(
                x, g, steps[-1].a, steps[-1].M, steps[-1].x_inf, steps[-1].g_inf
            )
            steps.append(
                Iteration(
                    g, x, a, L, g_gen, p_gen, x_gen, x_logits, x_inf, g_inf, p_inf
                )
            )
        steps = steps[1:]
        return steps

    def iteration(self, x, locations, a_prev, x_prev, g_prev):

        raise NotImplementedError

    def inference(self, x):
        raise NotImplementedError

    def generative(self):
        raise NotImplementedError

    def loss(self):
        raise NotImplementedError

    def init_trainable(self):
        raise NotImplementedError


# TODO causal transformer

class LayerNorm(nn.Module):
    # in this case, use fixed weights (i.e. z-score)
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        #self.gamma = nn.Parameter(torch.ones(d_model))
        #self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, p):
        mean = p.mean(-1, keepdim=True)
        var = p.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension. 

        out = (p - mean) / torch.sqrt(var + self.eps)
        #out = self.gamma * out + self.beta
        return out

# Might need to rename since these are essentially g
class PositionalEncoding(nn.Module):
    def __init__(self, params):
        super().__init__()
        dropout = params["dropout"]
        max_len = params["max_len"]
        d_model = params["d_model"]

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout)
        self.w_a = nn.Linear(
            max_len, d_model
        )  # dim was (max_len, 1, d_model) in original, may need to reshape
        # I think this also needs nonlinearity but it doesn't say what in the paper
        # look for other examples in the original TEM code of these learnable matrices
        # g_t+1 = f(g_t * W_a) where W_a is learnable action-dependent matrix

        # the thing is this is listed as an RNN but it seems simpler than that

    def forward(self, g: Tensor) -> Tensor:
        g = self.softmax(self.w_a(g))   # softmax before or after dropout?
        return self.dropout(g)  # do we still want a dropout?


class TransformerEmbedding(nn.Module):
    def __init__(self):
        # combine a positional encoding w a token embedding
        # paper mentions some embedding but not sure
        super(TransformerEmbedding, self).__init__()
        self.pos_emb = PositionalEncoding(params)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, p: Tuple):
        # position (e) initialized as 0 and then recurrently generated, independent of x?
        
        pos_emb = self.pos_emb(p[1])
        # TODO do dropout on both or neither?
        return (p[0], pos_emb)
       #return self.drop_out((tok_emb, pos_emb))


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

    # TODO causal masking
    # e_tild = q, k and x_tild = v
    def forward(self, g, e_tild, x_tild, mask=None):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        # this will come from params I think
        batch_size, head, length, d_tensor = 0, 0, 0, 0 #k.size()

        e_tild_t = e_tild.transpose(2, 3)
        score = (g @ e_tild_t) / math.sqrt(d_tensor)

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        x_tild = score @ x_tild

        return x_tild, score


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        # in TEM-t, keys and queries are treated as the same K, Q = EW_e
        self.w_e = nn.Linear(d_model, d_model)
        self.w_x = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, p: Tuple[Tensor, Tensor], mask=None):
        # 1. dot product with weight matrices
        # split H into E and X
        g = p[1]
        e_tild = self.w_e(p[1])
        x_tild = self.w_x(p[0])

        # 2. split tensor by number of heads
        #q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(g, e_tild, x_tild, mask=mask)

        # 4. concat and pass to linear layer
        #out = self.concat(out)
        #out = self.w_concat(out)

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
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        # TODO this is dumb replace
        # self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, p, src_mask):
        # 1. compute self attention
        _p = p
        p = self.attention(p, mask=src_mask)

        # 2. add and norm
        p[1] = self.dropout1(p[1])
        p[1] = self.norm1(p[1] + _p[1]) # just mentions norming on positional encoding

        # 3. positionwise feed forward network
        _p = p
        p = self.ffn(p)

        # 4. add and norm
        p[1] = self.dropout2(p[1])
        p[1] = self.norm2(p[1] + _p[1])
        return p


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

    # p input as defined by outer product in paper
    def forward(self, p, src_mask):
        p = self.emb(p)

        for layer in self.layers:
            p = layer(p, src_mask)

        return p

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, p):
        p = self.linear1(p)
        p = self.relu(p)
        p = self.dropout(p)
        p = self.linear2(p)
        return p

class DecoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, dec, enc, trg_mask, src_mask):
        # 1. compute self attention
        _p = dec
        p = self.self_attention(dec, mask=trg_mask)
        
        # 2. add and norm
        p = self.dropout1(p)
        p = self.norm1(p + _p)

        if enc is not None:
            # 3. compute encoder - decoder attention
            _p = p
            p = self.enc_dec_attention(p, mask=src_mask)
            
            # 4. add and norm
            p = self.dropout2(p)
            p = self.norm2(p + _p)

        # 5. positionwise feed forward network
        _p = p
        p = self.ffn(p)
        
        # 6. add and norm
        p = self.dropout3(p)
        p = self.norm3(p + _p)
        return p


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
