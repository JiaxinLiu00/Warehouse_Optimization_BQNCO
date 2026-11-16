import numpy as np
import torch
from bqnco_framework.encoder import EncoderLayer
from bqnco_framework.attention import MultiHeadAttention
from bqnco_framework.role_policy import RolePolicy, IndependentPolicies, Standardizer, collate
from olsf_trs.mappo_models import make_models, IndependentCritics, RecurrentCritic
from olsf_trs.config import SimulationConfig
from olsf_trs.simulator import Warehouse, CONT_DIM, MSG_DIM


def _decisions(count=6, **kwargs):
    env = Warehouse(SimulationConfig(seed=5, **kwargs))
    out = []
    while (d := env.decision()) is not None and len(out) < count:
        out.append(d); env.step(0)
    return out


def test_ippo_switch():
    actor, critic = make_models("IPPO", width=24, heads=3, layers=1)
    assert isinstance(actor, IndependentPolicies) and isinstance(critic, IndependentCritics)
    assert all(not policy.communicate for policy in actor.actors)
    actor, critic = make_models("MAPPO", width=24, heads=3, layers=1)
    assert isinstance(actor, RolePolicy) and actor.communicate and isinstance(critic, RecurrentCritic)


def test_bqnco_encoder_communication_and_gradients():
    torch.manual_seed(3)
    actor = RolePolicy()
    assert all(isinstance(layer, EncoderLayer) for layer in actor.encoder)
    assert len(actor.encoder) == 3 and actor.communication.num_heads == 12
    assert actor.encoder[0].res_norm1.batchnorm is False          # ReZero residuals
    decision = next(d for d in _decisions(20) if len(d.actions) > 1)
    batch = collate([decision])
    batch["messages"] = batch["messages"].clone().requires_grad_(True)
    optimizer = torch.optim.Adam(actor.parameters(), lr=.001)
    before = actor.encoder[0].attn.lambda1.detach().clone()
    for _ in range(3):
        optimizer.zero_grad()
        logits = actor(batch)
        torch.nn.functional.cross_entropy(logits, torch.zeros(1, dtype=torch.long)).backward()
        optimizer.step()
    role = int(batch["roles"][0])
    others = [r for r in range(3) if r != role]
    assert batch["messages"].grad[0, others].abs().sum() > 0     # attention-based inter-agent communication
    assert not torch.equal(before, actor.encoder[0].attn.lambda1)


def test_attention_uneven_heads_and_mask():
    attn = MultiHeadAttention(128, 12, use_biases=False)
    x = torch.randn(2, 4, 128)
    assert attn(x, x).shape == x.shape
    mask = torch.zeros(2, 4, 4, dtype=torch.bool)
    mask[:, 0, :] = True
    changed = x.clone(); changed[:, 0] += 100
    assert torch.allclose(attn(x, x, mask)[:, 1:], attn(changed, changed, mask)[:, 1:], atol=1e-5)


def test_padding_invariance_of_batched_policy():
    torch.manual_seed(0)
    decisions = _decisions(8)
    for actor in (RolePolicy(width=32, heads=4, layers=2), IndependentPolicies(width=32, heads=4, layers=1)):
        actor.eval()
        with torch.no_grad():
            batched = actor(collate(decisions))
            for i, d in enumerate(decisions):
                alone = actor(collate([d]))[0]
                n = len(d.actions)
                assert torch.allclose(batched[i, :n], alone, atol=1e-5)
                assert torch.isinf(batched[i, n:]).all()


def test_standardizer_and_categorical_embeddings():
    rows = np.random.default_rng(1).normal(5, 3, size=(500, 4)); rows[:, 3] = 7.0
    norm = Standardizer(4); norm.fit(rows)
    z = norm(torch.as_tensor(rows, dtype=torch.float32))
    assert torch.allclose(z[:, :3].mean(0), torch.zeros(3), atol=1e-4)
    assert torch.allclose(z[:, :3].std(0, unbiased=False), torch.ones(3), atol=1e-3)
    assert torch.allclose(z[:, 3], torch.zeros(500)) and bool(norm.fitted)
    actor = RolePolicy(width=16, heads=2, layers=1)
    batch = collate(_decisions(1))
    base = actor(batch)
    batch["cat"] = batch["cat"].clone(); batch["cat"][..., 1] = (batch["cat"][..., 1] + 1) % 4
    assert not torch.allclose(base, actor(batch))                  # priorities enter through learned embeddings
    assert batch["cont"].shape[-1] == CONT_DIM and batch["messages"].shape[-1] == MSG_DIM
