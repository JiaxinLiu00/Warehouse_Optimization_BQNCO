"""Recurrent critics and the model factory."""
import numpy as np
import torch
from torch import nn
from olsf_trs.simulator import GLOBAL_DIM, MSG_DIM
from bqnco_framework.role_policy import RolePolicy, IndependentPolicies, Standardizer


class RecurrentCritic(nn.Module):
    def __init__(self, input_dim=GLOBAL_DIM, width=128):
        super().__init__()
        self.norm = Standardizer(input_dim)
        self.spatial = nn.Sequential(nn.Linear(input_dim, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())
        self.gru = nn.GRU(width, width, batch_first=True)
        self.value = nn.Linear(width, 1)

    def fit_standardizer(self, rows):
        self.norm.fit(rows)

    def forward(self, sequence, hidden=None):
        sequence = torch.as_tensor(np.asarray(sequence, dtype=np.float32) if not torch.is_tensor(sequence) else sequence,
                                   dtype=torch.float32, device=next(self.parameters()).device)
        if sequence.ndim == 2:
            sequence = sequence.unsqueeze(0)
        recurrent, hidden = self.gru(self.spatial(self.norm(sequence)), hidden)
        return self.value(recurrent).squeeze(-1), hidden


class IndependentCritics(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.critics = nn.ModuleList([RecurrentCritic(MSG_DIM, width) for _ in range(3)])

    def fit_standardizer(self, messages):
        messages = np.asarray(messages, dtype=np.float32).reshape(-1, 3, MSG_DIM)
        for r, critic in enumerate(self.critics):
            critic.fit_standardizer(messages[:, r])

    def forward(self, messages, roles, hidden=None):
        device = next(self.parameters()).device
        messages = torch.as_tensor(np.asarray(messages, dtype=np.float32), device=device)
        roles = torch.as_tensor(np.asarray(roles), dtype=torch.long, device=device)
        hidden = [None] * 3 if hidden is None else hidden
        values, new_hidden = [], []
        for r, critic in enumerate(self.critics):
            v, h = critic(messages[:, r, :], hidden[r])
            values.append(v.squeeze(0)); new_hidden.append(h)
        return torch.stack(values, dim=1).gather(1, roles[:, None]).squeeze(1), new_hidden


def critic_values(critic, decisions, algorithm, hidden=None):
    if algorithm == "MAPPO":
        values, h = critic(np.stack([d.global_state for d in decisions]), hidden)
        return values.squeeze(0), h
    return critic(np.stack([d.messages for d in decisions]), [d.role for d in decisions], hidden)


def fit_critic(critic, decisions, algorithm):
    if algorithm == "MAPPO":
        critic.fit_standardizer(np.stack([d.global_state for d in decisions]))
    else:
        critic.fit_standardizer(np.stack([d.messages for d in decisions]))


def make_models(algorithm="MAPPO", width=128, heads=12, layers=3):
    if algorithm == "MAPPO":
        return RolePolicy(width, heads, layers), RecurrentCritic(width=width)
    if algorithm == "IPPO":
        return IndependentPolicies(width, heads, layers), IndependentCritics(width)
    raise ValueError("Algorithm must be MAPPO or IPPO")
