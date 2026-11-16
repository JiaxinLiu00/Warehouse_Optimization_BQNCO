from dataclasses import replace
from types import SimpleNamespace
import copy
import numpy as np
import pytest
from olsf_trs.config import SimulationConfig, generate_instance
from olsf_trs.simulator import Warehouse
from olsf_trs.baselines import choose, movement_cover
from olsf_trs.mappo_training import role_gae
from olsf_trs.evaluation import summarize_runs, RUN_METRICS


def test_same_time_bin_does_not_merge_different_robots():
    env = Warehouse(SimulationConfig(seed=0))
    while (d := env.decision()) is not None:
        if d.role == 2:
            for group in d.members:
                assert len({tuple(env.robot_positions[a[0]]) for a in group}) == 1
                assert len({env._routes[a[0]][0] for a in group}) == 1
            if env.time == pytest.approx(56.908795952997004):
                assert d.index_of((0,)) != d.index_of((2,))
                return
        env.step_action(choose(env, d, 'G3'))
    pytest.fail('Expected robot decision was not visited')


def test_identical_robot_actions_have_same_physical_transition():
    cfg = SimulationConfig(seed=0)
    data = generate_instance(cfg)
    data['robot_positions'][:] = data['robot_positions'][0]
    env = Warehouse(cfg, data)
    while (d := env.decision()) is not None:
        if d.role == 2:
            group = next(g for g in d.members if len(g) > 1)
            a, b = group[:2]
            first, second = copy.deepcopy(env), copy.deepcopy(env)
            first.step_action(a); second.step_action(b)
            assert first.time == second.time
            assert first.moves == second.moves
            assert np.array_equal(first.remaining, second.remaining)
            assert np.array_equal(first.inventory, second.inventory)
            swap = np.arange(env.n['robots']); swap[a[0]], swap[b[0]] = swap[b[0]], swap[a[0]]
            assert np.array_equal(first.robot_free, second.robot_free[swap])
            return
        env.step(0)
    pytest.fail('Expected a robot decision')


def test_ippo_gae_uses_own_values_and_interval_rewards():
    samples = [dict(decision=SimpleNamespace(role=r), value=v, movement_reward=w, local_reward=0.)
               for r,v,w in zip([0,1,2,0],[10.,100.,1000.,20.],[-1.,-2.,-3.,-4.])]
    adv, ret = role_gae(samples, {0:8.,1:16.,2:32.}, True, gamma=.5, lam=0)
    assert ret[[0,3]] == pytest.approx([-.25,-4.])
    samples[1]['value'] = -99999.; samples[2]['value'] = 99999.
    assert role_gae(samples, {}, True, gamma=.5, lam=0)[1][[0,3]] == pytest.approx(ret[[0,3]])
    _, truncated = role_gae(samples, {0:8.,1:16.,2:32.}, False, gamma=.5, lam=0)
    assert truncated[3] == pytest.approx(0.)
    assert truncated[1] == pytest.approx(-2.5)


def test_conditioned_cover_cannot_beat_unconditioned_optimum():
    groups = {(0,1):[0], (0,2):[1]}
    def cost(plan):
        return sum(2*y-e for y,e,z in plan.values()) + 3*(2-sum(z for y,e,z in plan.values()))
    baseline = cost(movement_cover(groups, {0:2}))
    conditioned = cost(movement_cover(groups, {0:2}, forced_group=(0,1)))
    assert baseline == pytest.approx(1.)
    assert conditioned == pytest.approx(3.)


def test_statistics_share_completed_seeds_and_count_failures():
    rows=[]
    for method in ['MAPPO','G3']:
        for seed in [1,2]:
            row=dict(instance='S-1',system='multi_tote',method=method,seed=seed,
                     finished=not(method=='G3' and seed==2),completion_rate=1. if seed==1 else .5)
            row.update({m: float(seed*10) for m in RUN_METRICS});rows.append(row)
    summary = summarize_runs(rows,['S-1'],['multi_tote'],['MAPPO','G3'])
    assert [r['complete_runs'] for r in summary] == [2,1]
    assert all(r['paired_runs']==1 and r['mean_tote_moves']==10 for r in summary)
    rows[0]['finished']=False
    empty=summarize_runs(rows,['S-1'],['multi_tote'],['MAPPO','G3'])
    assert all(r['paired_runs']==0 and r['mean_tote_moves'] is None for r in empty)
