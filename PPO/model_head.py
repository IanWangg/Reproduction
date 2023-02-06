from abc import ABC

import torch
import torch.nn as nn
import torch.distributions as D

# TODO : implement Softmax head, Identity head, Gaussian head

class BaseHead(object):
    def __init__(
        self,
        feature_dim,
        output_dim,
        first_act : bool=True, # indicate whether first activate feature
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.first_act = first_act
        
    def forward(self, feature):
        raise NotImplemented
    
    
class DeterministicHead(BaseHead):
    def __init__(
        self,
        feature_dim,
        output_dim,
        first_act : bool=True,
    ):
        super().__init__(
            feature_dim=feature_dim,
            output_dim=output_dim,
            first_act=first_act
        )
        
        self.fc = nn.Sequential(
            nn.Identity() if self.first_act else nn.ReLU(),
            nn.Linear(self.feature_dim, output_dim),
        )
        
    def forward(self, feature):
        return self.fc(feature)
    
    
class SoftmaxHead(DeterministicHead):
    def __init__(
        self,
        feature_dim,
        output_dim,
        first_act : bool=True,
    ):
        super().__init__(
            feature_dim=feature_dim,
            output_dim=output_dim,
            first_act=first_act
        )
        self.distribution_type = None
        self.distribution = None
        
    def forward(self, feature, reparam=False, deterministic=False):
        logits = self.fc(feature)
        self.distribution = D.Categorical(logits)
        
        if deterministic:
            return self.get_best_action()
        
        if reparam:
            return self.distribution.rsample()
        else:
            return self.distribution.sample()
    
    def entropy(self, action):
        return self.distribution.entropy(action)

    def log_prob(self, action):
        return self.distribution.log_prob(action)
    
    def get_best_action(self):
        return torch.argmax(self.distribution.probs, dim=1)

# Gaussian head will automatically cast the output_dim to 2x,
# the first half is the mean, the second part is the std 
class DiagGaussianHead(StochasticHead):
    def __init__(
        self,
        feature_dim,
        output_dim,
        first_act : bool=True,
    ):
        super().__init__(
            feature_dim=feature_dim,
            output_dim=output_dim * 2,
            first_act=first_act,
        )
        
        self.distribution_type = D.Normal
        self.distribution = None
        
    def forward(self, feature, reparam=False, deterministic=False):
        logits = self.fc(feature)
        mean, std = torch.tensor_split(logits, 2, dim=1)
        self.distribution = self.distribution_type(mean, std)
        
        if deterministic:
            return self.get_best_action()
        
        if reparam:
            return self.distribution.rsample()
        else:
            return self.distribution.sample()
    
    def get_best_action(self):
        return self.distribution.mean