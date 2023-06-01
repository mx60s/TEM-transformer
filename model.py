#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a pytorch implementation of TEM-t (or TEM-transformer)
based on the paper here: https://arxiv.org/abs/2112.04035 and Jacob Bakerman's TEM Pytorch implementation here: https://github.com/jbakermans/torch_tem

I'm not sure if I'll actually finish this

@author: mx60s
"""
# Standard modules
import numpy as np
import torch
import pdb
import copy
from scipy.stats import truncnorm
# Custom modules
import utils

class Model(torch.nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()
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

class Iteration:
    def __init__(self, g = None, x = None, a = None, L = None, M = None, g_gen = None, p_gen = None, x_gen = None, x_logits = None, x_inf = None, g_inf = None, p_inf = None):
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
        accuracy = [np.argmax(prediction, axis=-1) == np.argmax(observation, axis=-1) for prediction in predictions]
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
        