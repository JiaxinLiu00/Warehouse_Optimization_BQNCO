"""Role-aware policies built on the BQ-NCO encoder."""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from olsf_trs.simulator import CONT_DIM, MSG_DIM, CAT_VOCAB, CAT_FIELDS
from bqnco_framework.attention import MultiHeadAttention
from bqnco_framework.encoder import EncoderLayer


class Standardizer(nn.Module):
    """Standardizes continuous attributes with statistics fitted on training data."""

    def __init__(self, dim):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        self.register_buffer("fitted", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def fit(self, rows):
        x = torch.as_tensor(np.asarray(rows, dtype=np.float32).reshape(-1, self.mean.numel()))
        if not len(x):
            return
        std = x.std(0, unbiased=False)
        self.mean.copy_(x.mean(0))
        self.std.copy_(torch.where(std > 1e-6, std, torch.ones_like(std)))
        self.fitted.fill_(True)

    def forward(self, x):
        return (x - self.mean) / self.std


class MaskedEncoderLayer(EncoderLayer):
    """BQ-NCO encoder layer with a key mask for padded candidate classes."""

    def forward(self, state, mask=None):
        state = self.res_norm1(state, self.attn(state, state, mask))
        return self.res_norm2(state, self.linear2(self.activation(self.linear1(state))))


def collate(decisions, device="cpu"):
    """Pad a list of decisions to a batch of candidate classes."""
    size = max(len(d.actions) for d in decisions)
    count = len(decisions)
    cont = np.zeros((count, size, CONT_DIM), dtype=np.float32)
    cat = np.zeros((count, size, len(CAT_FIELDS)), dtype=np.int64)
    mask = np.zeros((count, size), dtype=bool)
    for i, d in enumerate(decisions):
        n = len(d.actions)
        cont[i, :n] = d.cont; cat[i, :n] = d.cat; mask[i, :n] = True
    messages = np.stack([d.messages for d in decisions]).astype(np.float32)
    roles = np.array([d.role for d in decisions], dtype=np.int64)
    return dict(cont=torch.as_tensor(cont, device=device), cat=torch.as_tensor(cat, device=device),
                mask=torch.as_tensor(mask, device=device), messages=torch.as_tensor(messages, device=device),
                roles=torch.as_tensor(roles, device=device))


def masked_log_softmax(logits, mask):
    return torch.log_softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)


def masked_entropy(logits, mask):
    logp = masked_log_softmax(logits, mask)
    # masked_fill keeps gradients finite for padded entries
    return -(logp.exp() * logp.masked_fill(~mask, 0.0)).sum(-1)


class RolePolicy(nn.Module):
    def __init__(self, width=128, heads=12, layers=3, communicate=True):
        super().__init__()
        self.communicate = communicate
        self.cont_norm = Standardizer(CONT_DIM)
        self.msg_norm = Standardizer(MSG_DIM)
        self.input = nn.Linear(CONT_DIM, width)
        self.categorical = nn.ModuleList([nn.Embedding(v, width) for v in CAT_VOCAB])
        self.role_embedding = nn.Embedding(3, width)
        self.encoder = nn.ModuleList([MaskedEncoderLayer(heads, "softmax", width, 4 * width, "relu", 0.0, False)
                                      for _ in range(layers)])
        self.message_encoder = nn.Sequential(nn.Linear(MSG_DIM, width), nn.ReLU())
        self.communication = MultiHeadAttention(width, heads, scale_dot_att=True)
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(2 * width, width), nn.ReLU(), nn.Linear(width, 1))
                                    for _ in range(3)])

    @property
    def device(self):
        return next(self.parameters()).device

    def fit_standardizers(self, cont_rows, msg_rows):
        self.cont_norm.fit(cont_rows); self.msg_norm.fit(msg_rows)

    def standardize(self, cont, roles=None):
        return self.cont_norm(cont)

    def forward(self, batch):
        cont, cat, mask, roles = batch["cont"], batch["cat"], batch["mask"], batch["roles"]
        count, size, _ = cont.shape
        h = self.input(self.cont_norm(cont))
        for field, embedding in enumerate(self.categorical):
            h = h + embedding(cat[..., field])
        h = torch.relu(h) + self.role_embedding(roles)[:, None, :]
        key_mask = (~mask)[:, :, None].expand(count, size, size)
        for layer in self.encoder:
            h = layer(h, key_mask)
        m = self.message_encoder(self.msg_norm(batch["messages"])) + self.role_embedding.weight[None]
        if self.communicate:
            m = m + self.communication(m, m)
        context = m[torch.arange(count, device=m.device), roles]
        z = torch.cat([h, context[:, None, :].expand(count, size, -1)], dim=-1)
        logits = torch.stack([head(z).squeeze(-1) for head in self.heads], dim=-1)
        logits = logits.gather(-1, roles[:, None, None].expand(count, size, 1)).squeeze(-1)
        return logits.masked_fill(~mask, float("-inf"))

    def distribution(self, decision):
        return Categorical(logits=self(collate([decision], self.device))[0])


class IndependentPolicies(nn.Module):
    """IPPO: separate actor parameters per role and only the role's own observation."""

    def __init__(self, width=128, heads=12, layers=3):
        super().__init__()
        self.actors = nn.ModuleList([RolePolicy(width, heads, layers, False) for _ in range(3)])

    @property
    def device(self):
        return next(self.parameters()).device

    def fit_standardizers(self, cont_rows, msg_rows):
        for actor in self.actors:
            actor.fit_standardizers(cont_rows, msg_rows)

    def standardize(self, cont, roles=None):
        return self.actors[0].cont_norm(cont)

    def forward(self, batch):
        roles = batch["roles"]
        out = torch.full(batch["mask"].shape, float("-inf"), device=roles.device)
        for r, actor in enumerate(self.actors):
            idx = torch.nonzero(roles == r).squeeze(1)
            if len(idx):
                out = out.index_put((idx,), actor({k: v[idx] for k, v in batch.items()}))
        return out

    def distribution(self, decision):
        return Categorical(logits=self(collate([decision], self.device))[0])
