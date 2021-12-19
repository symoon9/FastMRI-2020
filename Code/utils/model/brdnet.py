import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BRDNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(BRDNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding='valid')
        self.relu = nn.LeakyReLU(inplace=True)

class BatchRenormalization(nn.Module):
    def __init__(self, epsilon=1e-3, mode=0, axis=-1, momentum=0.99,
                 r_max_value=3., d_max_value=5., t_delta=1e-3, weights=None, beta_init='zero',
                 gamma_init='one', gamma_regularizer=None, beta_regularizer=None,
                 **kwargs):
        super(BatchRenormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.beta_init = initializers.get(beta_init)
        self.gamma_init = initializers.get(gamma_init)
        self.epsilon = epsilon
        self.mode = mode
        self.axis = axis
        self.momentum = momentum
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.initial_weights = weights
        self.r_max_value = r_max_value
        self.d_max_value = d_max_value
        self.t_delta = t_delta
        if self.mode == 0:
            self.uses_learning_phase = True
