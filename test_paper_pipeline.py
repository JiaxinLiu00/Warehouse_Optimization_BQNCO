from dataclasses import replace
import csv
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch
from olsf_trs.config import SimulationConfig, configurations, generate_instance, instance_fingerprint, storage_layout
from olsf_trs.simulator import Warehouse, KIND_STOP
from olsf_trs.mappo_models import RecurrentCritic, make_models
from olsf_trs.baselines import (choose, expert_action, minimum_tote_cover, movement_cover, order_assignment_plan,
                                robot_assignment, resolve_solver)
from olsf_trs.mappo_training import gae, train, pretrain, load_checkpoint, rollout, local_reward, DEFAULT_LAMBDAS
from olsf_trs.evaluation import evaluate, gap_evaluation
from bqnco_framework.role_policy import RolePolicy, collate

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]


def run(env, policy):
    while (d := env.decision()) is not None:
        env.step_action(policy(env, d))
    return env


def test_manuscript_sizes_layout_and_seeded_arrivals():
    assert len(configurations()) == 24
    assert configurations()['L-15'] == dict(skus=600, orders=350, robots=70, workstations=25, totes=5000)
    cfg = SimulationConfig(instance='L-15', seed=7)
    a = generate_instance(cfg); b = generate_instance(cfg)
    assert instance_fingerprint(a) == instance_fingerprint(b)
    assert instance_fingerprint(a) != instance_fingerprint(generate_instance(replace(cfg, seed=8)))
    assert np.all(np.diff(a['arrivals']) > 0)
    assert np.all(np.bincount(a['tote_sku'], weights=a['inventory'], minlength=600) >= a['demand'].sum(0))
    rack = generate_instance(replace(cfg, system='rack_climbing'))
    for key in ('demand', 'tote_positions', 'robot_positions', 'arrivals'):
        assert np.array_equal(a[key], rack[key])                  # paired instances across architectures
    cells = {tuple(c) for c in storage_layout(cfg)['cells']}
    assert len({tuple(p) for p in a['tote_positions']}) == 5000 and all(tuple(p) in cells for p in a['tote_positions'])
    assert np.all(a['robot_positions'][:, 2] == 0)
    means = [np.diff(np.r_[0, generate_instance(replace(cfg, seed=i))['arrivals']]).mean() for i in range(50)]
    assert np.mean(means) == pytest.approx(600 / 350, rel=.05)       # Poisson rate N_o/10 per minute


@pytest.mark.parametrize('system', ['multi_tote', 'rack_climbing'])
@pytest.mark.parametrize('method', ['R3', 'G3', 'C-SGH', 'expert'])
def test_end_to_end_conservation(system, method):
    env = Warehouse(SimulationConfig(system=system, seed=2), policy_roles=())
    initial = env.inventory.sum(); demand = env.remaining.sum(); reward = 0.0
    while (d := env.decision()) is not None:
        assert np.isfinite(d.cont).all() and len(d.actions) > 0 and not d.quotiented
        action = expert_action(env, d, "scipy") if method == "expert" else choose(env, d, method)
        _, r, _ = env.step_action(action); reward += r
        assert env.decisions < 2000
        assert (env.inventory >= 0).all() and (env.remaining >= 0).all()
        assert len(env.cargo) <= env.config.robot_capacity
        for s in range(env.config.size['workstations']):
            assert len(env.active(s)) <= env.config.slots_per_station
    assert env.complete.all() and env.robot_free.all()
    assert initial - env.inventory.sum() == demand
    assert reward == -env.moves
    assert env.retrievals >= env.returns > 0


def test_bisimulation_classes_merge_only_equal_attributes():
    cfg = SimulationConfig(instance='S-9', seed=11)
    env = Warehouse(cfg)
    seen = 0
    while (d := env.decision()) is not None:
        assert d.quotiented and d.classes <= d.raw_candidates and len(d.actions) == d.classes
        assert np.allclose(d.cont[:, -1], np.log(d.counts))
        if d.role == 1 and env.cargo:
            assert d.cat[-1, 0] == KIND_STOP                         # stop is always available
        seen += int(any(len(m) > 1 for m in d.members))
        env.step(0)
    assert env.metrics()['bq_candidate_reduction'] >= 0
    raw = Warehouse(replace(cfg, quotient=False))
    d = raw.decision()
    assert all(len(m) == 1 for m in d.members) and np.allclose(d.cont[:, -1], 0)
    exact = Warehouse(replace(cfg, bq_time_bin_s=0, bq_distance_bin_m=0))
    exact_d, binned_d = exact.decision(), Warehouse(cfg).decision()
    assert exact_d.classes >= binned_d.classes                      # exact equality never merges more


def test_policy_scores_every_candidate_class():
    env = Warehouse(SimulationConfig(instance='L-1', seed=200042, stochastic_arrivals=False))
    d = env.decision()
    assert d.role == 0 and d.classes > 64                    # no candidate window before the policy
    assert len(d.actions) == len(d.members) == d.classes
    assert sum(len(m) for m in d.members) == d.raw_candidates


def test_invalid_action():
    env = Warehouse(SimulationConfig(stochastic_arrivals=False))
    with pytest.raises(ValueError):
        env.step(-1)
    with pytest.raises(ValueError):
        env.step_action((999, 0))


def test_exact_movement_cover_matches_enumeration():
    rng = np.random.default_rng(0)
    for _ in range(100):
        n = int(rng.integers(1, 7)); sku = rng.integers(0, 2, n); inv = rng.integers(1, 5, n)
        demand = {k: int(rng.integers(0, 6)) for k in range(2)}
        groups = {}
        for t in range(n):
            groups.setdefault((int(sku[t]), int(inv[t])), []).append(t)
        plan = movement_cover(groups, demand, "scipy")
        covered = {k: sum(z for (s, _), (_, _, z) in plan.items() if s == k) for k in demand}
        objective = sum(2 * y - e for y, e, _ in plan.values()) + 3.0 * sum(d - covered[k] for k, d in demand.items())
        best = math.inf
        for mask in itertools.product([0, 1], repeat=n):
            cost = 0.0
            for k, d in demand.items():
                left = d
                for s in sorted([inv[t] for t in range(n) if mask[t] and sku[t] == k], reverse=True):
                    use = min(s, left); left -= use; cost += 1 if use == s else 2
                cost += 3.0 * left
            best = min(best, cost)
        assert objective == pytest.approx(best)
    answer = minimum_tote_cover([2, 3, 1, 4], [0, 0, 1, 1], [4, 3])
    assert answer.sum() == 3
    with pytest.raises(ValueError):
        minimum_tote_cover([1], [0], [3])


def test_order_plan_fills_slots_with_feasible_pairs():
    env = Warehouse(SimulationConfig(instance='S-3', seed=4, stochastic_arrivals=False), policy_roles=())
    d = env.decision()
    assert d.role == 0
    plan = order_assignment_plan(env, "scipy")
    free = sum(env.config.slots_per_station - len(env.active(s)) for s in range(env.n['workstations']))
    assert len(plan) == min(free, int(env.released.sum()))
    assert len({o for o, _ in plan}) == len(plan) and all(env.feasible(0, p) for p in plan)


def test_robot_assignment_matches_enumeration():
    for system in ('multi_tote', 'rack_climbing'):
        env = Warehouse(SimulationConfig(instance='S-4', system=system, seed=9), policy_roles=())
        checked = 0
        while (d := env.decision()) is not None:
            if d.role == 2:
                chosen = robot_assignment(env, "scipy")[0]
                routes = env._robot_routes(np.flatnonzero(env.robot_free))
                assert routes[chosen][0] == pytest.approx(min(v[0] for v in routes.values()))
                checked += 1
            env.step_action(choose(env, d, 'G3'))
        assert checked > 3


def test_local_reward_terms():
    env = Warehouse(SimulationConfig(seed=3))
    roles = set()
    while (d := env.decision()) is not None:
        env.step(0)
        info = env.last_info
        roles.add(info['role'])
        assert np.isfinite(info['f']) and all(np.isfinite(v) for v in info['impacts'].values())
        expected = info['f'] + sum(DEFAULT_LAMBDAS[f"{info['role']},{j}"] * v for j, v in info['impacts'].items())
        assert local_reward(info) == pytest.approx(expected)
    assert roles == {0, 1, 2}


def test_network_shapes_recurrence_and_local_ippo():
    env = Warehouse(SimulationConfig(seed=1)); d = env.decision()
    actor = RolePolicy()
    assert actor(collate([d])).shape == (1, len(d.actions))
    critic = RecurrentCritic()
    seq = torch.randn(1, 4, critic.norm.mean.numel()); v, h = critic(seq)
    h1 = None; parts = []
    for i in range(4):
        q, h1 = critic(seq[:, i:i + 1, :], h1); parts.append(q)
    assert torch.allclose(v, torch.cat(parts, dim=1), atol=1e-6)
    assert not torch.allclose(critic(seq[:, 3:4], None)[0], v[:, 3:4])
    ippo, _ = make_models('IPPO', width=24, heads=3, layers=1)
    batch = collate([d]); first = ippo(batch)
    others = [r for r in range(3) if r != d.role]
    batch['messages'][:, others] = 10
    assert torch.equal(first, ippo(batch))                        # IPPO only observes its own role


def test_gae_terminal_vs_truncated():
    _, terminal = gae([-1, -2], [.5, .7], bootstrap=10, terminated=True, gamma=1, lam=1)
    _, truncated = gae([-1, -2], [.5, .7], bootstrap=10, terminated=False, gamma=1, lam=1)
    assert np.allclose(terminal, [-3, -2])
    assert np.allclose(truncated, [7, 8])


def test_pretrain_train_evaluate_gap(tmp_path):
    cfg = SimulationConfig(seed=3)
    network = dict(width=24, heads=3, layers=1)
    settings = pretrain(cfg, tmp_path / 'il', ['S-1', 'S-2'], ['multi_tote'], train_instances=3, val_instances=2,
                        epochs=2, network=network, solver='scipy', device='cpu')
    assert settings['train_decisions'] > 0 and settings['val_decisions'] > 0
    before, critic, obj = load_checkpoint(tmp_path / 'il/model.pt')
    assert bool(before.cont_norm.fitted) and bool(critic.norm.fitted) and obj['training']['stage'] == 'imitation'
    train(cfg, tmp_path / 'ppo', instances=['S-1', 'S-2'], episodes=2, epochs=1,
          initial_checkpoint=tmp_path / 'il/model.pt', device='cpu')
    after, _, meta = load_checkpoint(tmp_path / 'ppo/model.pt')
    assert meta['algorithm'] == 'MAPPO' and meta['training']['instances'] == ['S-1', 'S-2']
    assert any(not torch.equal(a, b) for a, b in zip(before.parameters(), after.parameters()))
    train(cfg, tmp_path / 'ippo', algorithm='IPPO', episodes=1, epochs=1, network=network, device='cpu')
    rows = evaluate(cfg, tmp_path / 'eval', ['S-1'], ['multi_tote'], [10003], ['G3', 'MAPPO', 'IPPO', 'EXACT'],
                    tmp_path / 'ppo/model.pt', tmp_path / 'ippo/model.pt', device='cpu', solver='scipy')
    assert len({r['instance_fingerprint'] for r in rows}) == 1
    assert all(r['finished'] and r['seed'] == 10003 for r in rows)
    gaps = gap_evaluation(cfg, tmp_path / 'gap', ['S-1'], ['multi_tote'], [10003], tmp_path / 'il/model.pt',
                          solver='scipy', device='cpu')
    assert [r['agent'] for r in gaps] == ['order', 'tote', 'robot']
    assert all(r['inference_ms'] is not None and r['agent_decisions'] > 0 for r in gaps)
    assert all(r['module_ms'] >= r['inference_ms'] for r in gaps)    # module time sums the role's decisions
    assert all(r['gap_pct'] >= -1e-6 for r in gaps)
    assert [r['objective'] for r in gaps][-1] == 'trip_seconds'
    with (tmp_path / 'eval/summary.csv').open() as f:
        summary = {r['method']: r for r in csv.DictReader(f)}
    assert float(summary['EXACT']['mean_tote_moves']) == [r for r in rows if r['method'] == 'EXACT'][0]['tote_moves']


@pytest.mark.skipif(resolve_solver('auto') != 'gurobi', reason='Gurobi licence not available')
def test_gurobi_and_highs_agree():
    rng = np.random.default_rng(5)
    for _ in range(20):
        n = int(rng.integers(2, 9)); sku = rng.integers(0, 3, n); inv = rng.integers(1, 6, n)
        groups = {}
        for t in range(n):
            groups.setdefault((int(sku[t]), int(inv[t])), []).append(t)
        demand = {k: int(rng.integers(0, 8)) for k in range(3)}
        a = movement_cover(groups, demand, 'scipy'); b = movement_cover(groups, demand, 'gurobi')
        assert sum(2 * y - e for y, e, _ in a.values()) == sum(2 * y - e for y, e, _ in b.values())
    scipy_env = run(Warehouse(SimulationConfig(instance='S-2', seed=8), policy_roles=()), lambda e, d: expert_action(e, d, 'scipy'))
    gurobi_env = run(Warehouse(SimulationConfig(instance='S-2', seed=8), policy_roles=()), lambda e, d: expert_action(e, d, 'gurobi'))
    assert abs(scipy_env.moves - gurobi_env.moves) <= 0.05 * scipy_env.moves
