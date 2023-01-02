import numpy as np
import argparse

import matplotlib.pyplot as plt
from scipy import sparse
import torch
import random
from tqdm import tqdm
import seaborn as sns
from collections import Counter
import os
import pickle
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from copy import deepcopy

def toy_function():
    return None

class MLP(nn.Module):
    def __init__(self, in_size, h_sizes, out_size):
        super(MLP, self).__init__()
        self.hidden = nn.ModuleList([nn.Linear(in_size, h_sizes[0])])
        for k in range(len(h_sizes) - 1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k + 1]))
        self.out = nn.Linear(h_sizes[-1], out_size)

    def forward(self, x):
        for layer in self.hidden:
            x = F.relu((layer(x)))
        output = self.out(x)
        return output


class GFN:
    def __init__(self, m, h_sizes, bs):
        self.m = m
        self.forward_mlp = MLP(m, h_sizes, 2 * m)
        self.backward_mlp = MLP(m, h_sizes, m)
        self.logZ = MLP(m, h_sizes, 1)
        self.bs = bs

    def forward(self, batch):
        # batch : size bs*m
        logits = self.forward_mlp(batch)  # logits shape : bs* (2*m)
        logits[:, : self.m][batch > -0.5] = -float("inf")
        logits[:, self.m :][batch > -0.5] = -float("inf")
        return logits

    def backward(self, batch):
        # batch : size k*m
        logits = self.backward_mlp(batch)
        logits[batch < -0.5] = -float("inf")
        return logits

    def sample_forward_traj(self, temperature=1):
        thetas = -torch.ones(self.bs, self.m)
        traj = [deepcopy(thetas)]
        # print(f"traj {traj}")
        for i in range(self.m):
            logits = self.forward(thetas)
            probs = torch.softmax(logits / temperature, 1)
            # print("probs ", probs.shape)
            ixs = probs.multinomial(1).squeeze()
            # print("ixs ", ixs.shape)
            for (j, ix) in enumerate(ixs):
                # print(ix%self.m, ix//self.m)
                thetas[j, ix % self.m] = ix // self.m  # 0 if ix chosen is <m, 1 else
            traj.append(deepcopy(thetas))
            # print(f"traj {traj}")
        # return torch.cat(traj, 0)
        return torch.cat(traj, axis=1).reshape(self.bs, self.m + 1, self.m)

    def sample_backward_traj(self, states, temperature=1):
        thetas = deepcopy(states)
        traj = [deepcopy(thetas)]
        for i in range(self.m):
            logits = self.backward(thetas)
            probs = torch.softmax(logits / temperature, 1)
            ixs = probs.multinomial(1)
            for (j, ix) in enumerate(ixs):
                thetas[j, ix] = -1
            traj.append(deepcopy(thetas))
        traj.reverse()
        return torch.cat(traj, axis=1).reshape(self.bs, self.m + 1, self.m)

    def get_log_pf(self, traj):
        # Ugly code, a refaire
        m, bs = self.m, self.bs
        # Get forward logits
        forward_logits = self.forward(traj[:, :-1].reshape(bs * m, m)).reshape(
            bs, m, 2 * m
        )
        # Get forward logprobs
        diffs = traj[:, 1:, :] - traj[:, :-1, :]
        ## Get xs
        xs = diffs.argmax(2)  # actual ixs chosen in the trajectory
        y = torch.Tensor(
            [
                diffs.reshape(bs * m, m)[i, j].item()
                for i, j in enumerate(xs.reshape(bs * m))
            ]
        ).long()
        y = ((y - 1) * m).reshape(bs, m)
        xs = xs + y
        ## Compute forward logprobs
        forward_logprobs = torch.Tensor(
            [
                forward_logits.reshape(bs * m, 2 * m)[i, j].item()
                for i, j in enumerate(xs.reshape(bs * m))
            ]
        ).reshape(bs, m)
        ## Normalize
        forward_denominators = torch.logsumexp(forward_logits, 2)
        forward_logprobs = forward_logprobs - forward_denominators
        return forward_logprobs

    def get_log_pb(self, traj):
        m, bs = self.m, self.bs
        diffs = traj[:, 1:, :] - traj[:, :-1, :]
        backward_logits = self.backward(traj[:, 1:].reshape(bs * m, m)).reshape(
            bs, m, m
        )
        xs = diffs.argmax(2)  # actual ixs chosen in the trajectory
        # Get backward logprobs

        backward_logprobs = torch.Tensor(
            [
                backward_logits.reshape(bs * m, m)[i, j].item()
                for i, j in enumerate(xs.reshape(bs * m))
            ]
        ).reshape(bs, m)

        backward_denominators = torch.logsumexp(backward_logits, 2)
        backward_logprobs = backward_logprobs - backward_denominators
        return backward_logprobs

    def get_NNLoss(self, traj):
        # assert traj[-1] in dataset
        log_pf = self.get_log_pf(traj).sum(axis=1)
        log_pb = self.get_log_pb(traj).sum(axis=1).detach()

        return -(log_pf - log_pb)

    def get_CLoss(self, traj1, traj2):
        log_pf1, log_pb1 = (
            self.get_log_pf(traj1).sum(axis=1).detach(),
            self.get_log_pb(traj1).sum(axis=1),
        )
        log_pf2, log_pb2 = (
            self.get_log_pf(traj2).sum(axis=1).detach(),
            self.get_log_pb(traj2).sum(axis=1),
        )
        return ((log_pf1 - log_pb1) - (log_pf2 - log_pb2)) ** 2

