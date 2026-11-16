"""Paired evaluation of policies and baselines, and small-scale agent-level gaps."""
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t
from olsf_trs.baselines import HEURISTICS, resolve_solver
from olsf_trs.mappo_training import (rollout, load_checkpoint, checkpoint_hash, seed_all, environment_metadata,
                               resolve_device)

METHODS = ("MAPPO", "IPPO") + HEURISTICS + ("EXACT",)
AGENTS = ("order", "tote", "robot")
RUN_METRICS = ("tote_moves", "retrievals", "returns", "runtime_s", "simulation_seconds",
               "policy_inference_mean_ms", "policy_inference_p95_ms", "bq_candidate_reduction")


def write_csv(path, rows):
    if rows:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with Path(path).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def describe(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    out = dict(mean=float(values.mean()) if len(values) else None,
               std=float(values.std(ddof=1)) if len(values) > 1 else None,
               ci95_halfwidth=(float(student_t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
                               if len(values) > 1 else None))
    return out


def _load(methods, checkpoint, ippo_checkpoint, device):
    models, hashes = {}, {}
    for method, path in (("MAPPO", checkpoint), ("IPPO", ippo_checkpoint)):
        if method in methods:
            if not path:
                raise ValueError(method + " requires its checkpoint argument")
            actor, _, obj = load_checkpoint(path, device)
            if obj["algorithm"] != method:
                raise ValueError("Checkpoint algorithm mismatch for " + method)
            models[method] = actor; hashes[method] = checkpoint_hash(path)
    return models, hashes


def evaluate(config, output, instances, systems, seeds, methods, checkpoint=None, ippo_checkpoint=None,
             max_decisions=1_000_000, device="auto", solver="auto", save_traces=False):
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise ValueError("Unknown methods: " + ", ".join(sorted(unknown)))
    device = resolve_device(device)
    solver = resolve_solver(solver) if "EXACT" in methods else None
    models, hashes = _load(methods, checkpoint, ippo_checkpoint, device)
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    manifest = dict(simulation_defaults=asdict(config), instances=list(instances), systems=list(systems),
                    seeds=list(seeds), methods=list(methods), checkpoint_hashes=hashes,
                    environment=environment_metadata(device, solver),
                    runtime_scope="decision loop including simulation and policy; excludes instance generation and checkpoint loading",
                    inference_scope="actor forward pass and action selection only; excludes critic",
                    baseline_scope="rule-based and exact methods act on the complete, unquotiented candidate set")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    rows = []
    for instance in instances:
        for system in systems:
            for seed in seeds:
                cfg = replace(config, instance=instance, system=system, seed=seed)
                for method in methods:
                    seed_all(seed)
                    if method in models:
                        _, _, metrics, trace = rollout(cfg, actor=models[method], algorithm=method, deterministic=True,
                                                       max_decisions=max_decisions, keep_trace=save_traces)
                    else:
                        _, _, metrics, trace = rollout(cfg, method="expert" if method == "EXACT" else method,
                                                       solver=solver or "scipy", max_decisions=max_decisions,
                                                       keep_trace=save_traces)
                    rows.append(dict(instance=instance, system=system, seed=seed, method=method, **metrics))
                    if save_traces:
                        (out / f"{instance}_{system}_{method}_{seed}_trace.json").write_text(json.dumps(trace))
                    write_csv(out / "runs.csv", rows)
                    print(f"{instance} {system} {method} seed={seed}: moves={metrics['tote_moves']} "
                          f"completion={metrics['completion_rate']:.3f}", flush=True)
    summary = summarize_runs(rows, instances, systems, methods)
    write_csv(out / "summary.csv", summary)
    return rows


def summarize_runs(rows, instances, systems, methods):
    summary = []
    for instance in instances:
        for system in systems:
            complete_sets = [{r["seed"] for r in rows if r["instance"] == instance and r["system"] == system
                              and r["method"] == method and r["finished"]} for method in methods]
            paired_seeds = set.intersection(*complete_sets) if complete_sets else set()
            for method in methods:
                group = [r for r in rows if (r["instance"], r["system"], r["method"]) == (instance, system, method)]
                complete = [r for r in group if r["finished"]]
                result = dict(instance=instance, system=system, method=method, runs=len(group),
                              complete_runs=len(complete), paired_runs=len(paired_seeds),
                              paired_seeds=";".join(map(str, sorted(paired_seeds))),
                              mean_completion_rate=float(np.mean([r["completion_rate"] for r in group])))
                for metric in RUN_METRICS:
                    for stat, value in describe([r[metric] for r in complete if r["seed"] in paired_seeds]).items():
                        result[f"{stat}_{metric}"] = value
                summary.append(result)
    return summary


def gap_evaluation(config, output, instances, systems, seeds, checkpoint, solver="auto", device="auto",
                   max_decisions=1_000_000):
    """Compare local objectives on states visited by a shared expert trajectory.

    Order: full slot assignment conditioned on one chosen pair.
    Tote: residual cover conditioned on one chosen tote at the expert's station.
    Robot: pickup trip time for a chosen robot. Stop-loading actions are outside
    the tote-cover subproblem and are counted separately.
    """
    import time
    import torch
    from .simulator import Warehouse
    from .baselines import expert_action, local_decision_costs
    device = resolve_device(device); solver = resolve_solver(solver)
    actor, _, obj = load_checkpoint(checkpoint, device)
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    objectives = {"order": "movements_plus_uncovered_unit_penalty",
                  "tote": "movements_plus_uncovered_unit_penalty", "robot": "trip_seconds"}
    manifest = dict(simulation_defaults=asdict(config), instances=list(instances), systems=list(systems),
                    seeds=list(seeds), checkpoint_sha256=checkpoint_hash(checkpoint), algorithm=obj["algorithm"],
                    environment=environment_metadata(device, solver), objectives=objectives,
                    reference="same-state local optimum; policy objective fixes its selected action",
                    tote_scope="next-tote selection at the expert station; excludes stop-loading decisions",
                    inference_scope="actor forward pass and local action selection; excludes local solvers")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    rows, decisions = [], []
    for instance in instances:
        for system in systems:
            for seed in seeds:
                seed_all(seed)
                env = Warehouse(replace(config, instance=instance, system=system, seed=seed))
                grouped = {r: [] for r in range(3)}; skipped = 0
                while (d := env.decision()) is not None and env.decisions < max_decisions:
                    expert = expert_action(env, d, solver)
                    if d.role == 1 and expert[0] < 0:
                        skipped += 1
                    else:
                        allowed = [i for i, a in enumerate(d.actions)
                                   if d.role != 1 or (a[0] >= 0 and a[1] == expert[1])]
                        if not allowed:
                            raise RuntimeError("Local choice set is empty")
                        if device.startswith("cuda"):
                            torch.cuda.synchronize()
                        tic = time.perf_counter()
                        with torch.no_grad():
                            logits = actor.distribution(d).logits
                            index = allowed[int(logits[allowed].argmax())]
                        if device.startswith("cuda"):
                            torch.cuda.synchronize()
                        elapsed = (time.perf_counter() - tic) * 1000
                        ref, selected, objective = local_decision_costs(env, d.actions[index], solver)
                        if selected < ref - 1e-5:
                            raise RuntimeError("Conditioned local optimum is below the reference")
                        row = dict(instance=instance, system=system, seed=seed, step=env.decisions,
                                   agent=AGENTS[d.role], objective=objective, reference_objective=ref,
                                   policy_objective=selected, excess=max(0.0, selected-ref), inference_ms=elapsed)
                        decisions.append(row); grouped[d.role].append(row)
                    env.step_action(expert)
                for role, agent in enumerate(AGENTS):
                    group = grouped[role]
                    ref = sum(x["reference_objective"] for x in group)
                    selected = sum(x["policy_objective"] for x in group)
                    excess = sum(x["excess"] for x in group)
                    rows.append(dict(instance=instance, system=system, seed=seed, agent=agent,
                                     objective=objectives[agent], reference_objective=ref, policy_objective=selected,
                                     gap_pct=100 * excess/ref if ref > 1e-9 else (0.0 if excess <= 1e-9 else None),
                                     inference_ms=float(np.mean([x["inference_ms"] for x in group])) if group else None,
                                     module_ms=sum(x["inference_ms"] for x in group), agent_decisions=len(group),
                                     skipped_stop_decisions=skipped if role==1 else 0, reference_finished=env.done))
    write_csv(out / "gap_decisions.csv", decisions)
    write_csv(out / "gap_runs.csv", rows)
    summary = []
    for instance in instances:
        for system in systems:
            for agent in AGENTS:
                all_rows = [r for r in rows if (r["instance"], r["system"], r["agent"]) == (instance, system, agent)]
                group = [r for r in all_rows if r["reference_finished"] and r["agent_decisions"] > 0]
                result = dict(instance=instance, system=system, agent=agent, objective=objectives[agent],
                              runs=len(all_rows), complete_runs=len(group),
                              gap_runs=sum(r["gap_pct"] is not None for r in group))
                for metric in ("gap_pct", "inference_ms", "module_ms", "reference_objective", "policy_objective"):
                    for stat, value in describe([r[metric] for r in group]).items():
                        result[f"{stat}_{metric}"] = value
                summary.append(result)
    write_csv(out / "gap_summary.csv", summary)
    return rows
