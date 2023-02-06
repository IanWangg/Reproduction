import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from typing import Tuple, List


class BaseBody(nn.Module):
    def __init__(
        self,
        input_shape : tuple,
        feature_dim : int,
    ):
        super().__init__()

        self.input_shape = input_shape
        self.feature_dim = feature_dim

    def forward(self, state):
        raise NotImplemented 

class RainbowBody(BaseBody):
    def __init__(
        self,
        input_shape : Tuple,
        feature_dim : int = 512,
    ):
        super().__init__(
            input_shape=input_shape,
            feature_dime=feature_dim,
        )

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
        )

        flatten_dim = self.conv(
            torch.zeros(*input_shape
        ).unsqueeze(0)).flatten().shape[0]
        
        self.feature_dim = flatten_dim

    def forward(self, state):
        conv_features = self.conv(state).flatten()
        conv_features = conv_features.view(state.shape[0], -1)
        return conv_features


class MlpBody(BaseBody):
    def __init__(
        input_shape : Tuple,
        feature_dim : int = 256,
    ):
        super().__init__(
            input_shape=input_shape,
            feature_dim=feature_dim,
        )

        self.fc = nn.Sequential(
            nn.Linear(input_shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
        )

    def forward(self, state):
        return self.fc(state)