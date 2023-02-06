from abc import ABC

import torch
import torch.nn as nn
import torch.distributions as D

from utils import TanhBijector, sum_independent_dims

# TODO : implement Softmax head, Identity head, Gaussian head

class DeterministicHead(nn.Module):
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
    
    def entropy(self):
        return sum_independent_dims(self.distribution.entropy())

    def log_prob(self, action):
        return sum_independent_dims(self.distribution.log_prob(action))
    
    def get_best_action(self):
        return torch.argmax(self.distribution.probs, dim=1)

# Gaussian head will automatically cast the output_dim to 2x,
# the first half is the mean, the second part is the std 
class DiagGaussianHead(DeterministicHead):
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
        mean, log_std = torch.tensor_split(logits, 2, dim=1)
        self.distribution = self.distribution_type(mean, torch.exp(log_std))
        
        if deterministic:
            return self.get_best_action()
        
        if reparam:
            return self.distribution.rsample()
        else:
            return self.distribution.sample()
        
    def entropy(self):
        return sum_independent_dims(self.distribution.entropy())

    def log_prob(self, action):
        return sum_independent_dims(self.distribution.log_prob(action))
    
    def get_best_action(self):
        return self.distribution.mean
    
class SquashDiagGaussianHead(DeterministicHead):
    def __init__(
        self,
        feature_dim,
        output_dim,
        max_action : int=1,
        first_act : bool=True,
    ):
        super().__init__(
            feature_dim=feature_dim,
            output_dim=output_dim * 2,
            first_act=first_act,
        )
        
        self.max_action = max_action
        self.distribution_type = D.Normal
        self.distribution = None
        
    def forward(self, feature, reparam=False, deterministic=False):
        logits = self.fc(feature)
        mean, log_std = torch.tensor_split(logits, 2, dim=1)
        self.distribution = self.distribution_type(mean, torch.exp(log_std))
        
        if deterministic:
            action = self.get_best_action()
        
        if reparam:
            action = self.distribution.rsample()
        else:
            action = self.distribution.sample()
        
        # return squashed action and original action
        return torch.tanh(action) * self.max_action
    
    # there is no analytical form for squashed diag guassian dist
    # can use - log_prob.mean() as approximation
    def entropy(self):
        return None

    def log_prob(self, action):
        return sum_independent_dims(self.distribution.log_prob(TanhBijector.inverse(action)))
    
    def get_best_action(self):
        return torch.tanh(self.distribution.mean)