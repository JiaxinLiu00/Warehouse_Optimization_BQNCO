"""Imitation pretraining, recurrent MAPPO/IPPO training, rollouts and model IO."""
from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import json
import math
import platform
import random
import time
import numpy as np
import torch
from olsf_trs import __version__
from olsf_trs.config import SPLIT_OFFSETS, expand_instances
from olsf_trs.simulator import Warehouse
from olsf_trs.mappo_models import make_models, critic_values, fit_critic
from olsf_trs.baselines import choose, expert_action, resolve_solver, solver_version
from bqnco_framework.role_policy import collate, masked_log_softmax, masked_entropy

DEFAULT_LAMBDAS = {"0,1": 0.5, "0,2": 0.5, "1,0": 0.5, "1,2": 0.5, "2,0": 0.5, "2,1": 0.5}
CHECKPOINT_FORMAT = 4


def resolve_device(device="auto"):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def environment_metadata(device="cpu", solver=None):
    import scipy
    meta = dict(python=platform.python_version(), torch=torch.__version__, torch_cuda=torch.version.cuda,
                numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform(),
                machine=platform.machine(), processor=platform.processor(), implementation_version=__version__,
                device=str(device), cuda_available=torch.cuda.is_available())
    if torch.cuda.is_available():
        meta["gpu"] = torch.cuda.get_device_name(0)
    if solver:
        meta["solver"] = solver_version(solver)
    return meta


def checkpoint_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_checkpoint(path, actor, critic, algorithm, network, config, training):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(format_version=CHECKPOINT_FORMAT, encoder="bqnco", algorithm=algorithm, network=network,
                    actor=actor.state_dict(), critic=critic.state_dict(), simulation=asdict(config),
                    training=training), path)


def load_checkpoint(path, device="cpu"):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    obj = torch.load(path, map_location=device, weights_only=True)
    if obj.get("format_version") != CHECKPOINT_FORMAT:
        raise ValueError("Unsupported checkpoint format")
    actor, critic = make_models(obj["algorithm"], **obj["network"])
    actor.load_state_dict(obj["actor"], strict=True)
    critic.load_state_dict(obj["critic"], strict=True)
    actor.to(device).eval(); critic.to(device).eval()
    return actor, critic, obj


def local_reward(info, lambdas=None):
    """R_i = f_i(s, a_i) + sum_{j != i} lambda_ij Impact_ij(a_i, a_j)."""
    if not info:
        return 0.0
    lambdas = DEFAULT_LAMBDAS if lambdas is None else lambdas
    i = info["role"]
    return float(info["f"] + sum(lambdas.get(f"{i},{j}", 0.0) * v for j, v in info["impacts"].items()))


def gae(rewards, values, bootstrap=0.0, terminated=True, gamma=.99, lam=.95):
    rewards = np.asarray(rewards, dtype=np.float32); values = np.asarray(values, dtype=np.float32)
    advantages = np.zeros_like(rewards); carry = 0.0
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t + 1 < len(values) else (0.0 if terminated else bootstrap)
        delta = rewards[t] + gamma * next_value - values[t]
        carry = delta + gamma * lam * carry
        advantages[t] = carry
    return advantages, advantages + values


def role_gae(samples, bootstrap, terminated=True, gamma=.99, lam=.95, local_weight=.1):
    advantages = np.zeros(len(samples), dtype=np.float32)
    returns = np.zeros_like(advantages)
    for role in range(3):
        indices = [i for i, sample in enumerate(samples) if sample["decision"].role == role]
        carry = 0.0
        for k in reversed(range(len(indices))):
            i = indices[k]
            end = indices[k + 1] if k + 1 < len(indices) else len(samples)
            duration = end - i
            reward = sum(gamma ** (j - i) * samples[j]["movement_reward"] for j in range(i, end))
            reward += local_weight * samples[i]["local_reward"]
            next_value = (samples[end]["value"] if end < len(samples)
                          else (0.0 if terminated else bootstrap[role]))
            delta = reward + gamma ** duration * next_value - samples[i]["value"]
            carry = delta + (gamma * lam) ** duration * carry
            advantages[i] = carry
            returns[i] = carry + samples[i]["value"]
    return advantages, returns


def rollout(config, actor=None, critic=None, algorithm="MAPPO", method="G3", policy_roles=(0, 1, 2),
            fallback="expert", deterministic=False, solver="scipy", local_weight=0.0, lambdas=None,
            max_decisions=1_000_000, keep_trace=False):
    """Run one episode. Roles in ``policy_roles`` use the actor; other roles use ``fallback``.

    Without an actor every role uses ``method`` (R3, G3, C-SGH or expert).
    """
    learned = set(policy_roles) if actor is not None else set()
    env = Warehouse(config, policy_roles=tuple(sorted(learned)))
    decision = env.decision()
    hidden = None
    samples, trace = [], []
    inference = {0: [], 1: [], 2: []}
    start = time.perf_counter()
    while decision is not None and env.decisions < max_decisions:
        role = decision.role
        before = env.metrics() if keep_trace else None
        if role in learned:
            tic = time.perf_counter()
            with torch.no_grad():
                dist = actor.distribution(decision)
                index = int(dist.logits.argmax()) if deterministic else int(dist.sample())
                log_prob = float(dist.log_prob(torch.tensor(index, device=dist.logits.device)))
            inference[role].append(time.perf_counter() - tic)
            value = 0.0
            if critic is not None:
                with torch.no_grad():
                    v, hidden = critic_values(critic, [decision], algorithm, hidden)
                value = float(v[0])
            action = decision.actions[index]
            nxt, reward, _ = env.step(index)
        else:
            controller = method if actor is None else fallback
            action = expert_action(env, decision, solver) if controller == "expert" else choose(env, decision, controller)
            nxt, reward, _ = env.step_action(action)
            index, log_prob, value = None, 0.0, 0.0
        local = local_reward(env.last_info, lambdas)
        if role in learned:
            samples.append(dict(decision=decision, action=index, log_prob=log_prob, value=value,
                                reward=reward + local_weight * local, movement_reward=reward, local_reward=local))
        if keep_trace:
            trace.append(dict(step=env.decisions, role=role, action=list(action), classes=len(decision.actions),
                              raw_candidates=decision.raw_candidates, time_before=before["simulation_seconds"],
                              time_after=env.time, movement_reward=reward, local_reward=local,
                              retrievals=env.retrievals, returns=env.returns, remaining_units=int(env.remaining.sum())))
        decision = nxt
    bootstrap = {r: 0.0 for r in range(3)} if algorithm == "IPPO" else 0.0
    if decision is not None and critic is not None:
        with torch.no_grad():
            if algorithm == "IPPO":
                for role in range(3):
                    v, _ = critic_values(critic, [replace(decision, role=role)], algorithm, hidden)
                    bootstrap[role] = float(v[0])
            else:
                v, _ = critic_values(critic, [decision], algorithm, hidden)
                bootstrap = float(v[0])
    metrics = env.metrics()
    every = [t for times in inference.values() for t in times]
    metrics.update(runtime_s=time.perf_counter() - start, truncated=decision is not None,
                   policy_inference_mean_ms=float(np.mean(every) * 1000) if every else None,
                   policy_inference_p95_ms=float(np.percentile(every, 95) * 1000) if every else None)
    for r, name in enumerate(("order", "tote", "robot")):
        metrics[f"inference_ms_{name}"] = float(np.mean(inference[r]) * 1000) if inference[r] else None
        metrics[f"inference_total_ms_{name}"] = float(np.sum(inference[r]) * 1000) if inference[r] else None
        metrics[f"decisions_{name}"] = len(inference[r])
    return samples, bootstrap, metrics, trace


def split_jobs(config, instances, systems, count, split):
    instances = expand_instances(instances)
    return [(instances[i % len(instances)], systems[(i // len(instances)) % len(systems)],
             config.seed + SPLIT_OFFSETS[split] + i) for i in range(count)]


def collect_demonstrations(config, jobs, solver="scipy", max_decisions=1_000_000):
    """Exact-solver trajectories in the policy representation; labels index the expert's class."""
    trajectories, records, missing = [], [], 0
    for instance, system, seed in jobs:
        env = Warehouse(replace(config, instance=instance, system=system, seed=seed))
        decision = env.decision(); steps = []
        while decision is not None and env.decisions < max_decisions:
            action = expert_action(env, decision, solver)
            label = decision.index_of(action)
            if label is None:
                missing += 1
            else:
                steps.append((decision, label))
            decision, _, _ = env.step_action(action)
        trajectories.append(steps)
        records.append(dict(instance=instance, system=system, seed=seed, **env.metrics()))
    return trajectories, records, missing


def _imitation_batch(actor, decisions, labels, device, coupling_weight):
    batch = collate(decisions, device)
    labels = torch.as_tensor(labels, dtype=torch.long, device=device)
    logits = actor(batch)
    logp = masked_log_softmax(logits, batch["mask"])
    ce = -logp.gather(1, labels[:, None]).squeeze(1)
    features = actor.standardize(batch["cont"])
    predicted = (logp.exp()[..., None] * features).sum(1)          # I_pred: expected action vector
    true = features[torch.arange(len(labels), device=device), labels]   # I_true: expert action vector
    coupling = ((predicted - true) ** 2).sum(-1)
    return ce, coupling, logits.argmax(1) == labels, batch["roles"]


def pretrain(config, output, instances=("all-small",), systems=("multi_tote",), train_instances=1000,
             val_instances=300, epochs=30, network=None, coupling_weight=.1, solver="auto", device="auto",
             batch_size=64, learning_rate=3e-4, subsequences=2, algorithm="MAPPO", max_decisions=1_000_000):
    if train_instances < 1 or epochs < 1 or batch_size < 1:
        raise ValueError("Counts must be positive")
    device = resolve_device(device); solver = resolve_solver(solver)
    seed_all(config.seed)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    systems = list(systems)
    tic = time.perf_counter()
    train_traj, train_records, train_missing = collect_demonstrations(
        config, split_jobs(config, instances, systems, train_instances, "train"), solver, max_decisions)
    val_traj, val_records, val_missing = collect_demonstrations(
        config, split_jobs(config, instances, systems, val_instances, "validation"), solver, max_decisions)
    collection_s = time.perf_counter() - tic
    network = network or dict(width=128, heads=12, layers=3)
    actor, critic = make_models(algorithm, **network)
    actor.to(device); critic.to(device)
    decisions = [d for steps in train_traj for d, _ in steps]
    actor.fit_standardizers(np.concatenate([d.cont for d in decisions]), np.stack([d.messages for d in decisions]))
    fit_critic(critic, decisions, algorithm)
    optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    settings = dict(stage="imitation", algorithm=algorithm, instances=expand_instances(instances), systems=systems,
                    train_instances=train_instances, val_instances=val_instances, epochs=epochs,
                    coupling_weight=coupling_weight, solver=solver, batch_size=batch_size,
                    learning_rate=learning_rate, subsequences=subsequences,
                    train_decisions=len(decisions), val_decisions=sum(len(s) for s in val_traj),
                    unlabeled_decisions=dict(train=train_missing, validation=val_missing),
                    demonstration_seconds=collection_s)
    environment = environment_metadata(device, solver)
    (output / "demonstrations.csv").write_text(_csv([dict(split="train", **r) for r in train_records]
                                                    + [dict(split="validation", **r) for r in val_records]))
    rng = random.Random(config.seed)
    best = None; history = []
    for epoch in range(epochs):
        index = [(i, j) for i, steps in enumerate(train_traj) for j in range(len(steps))]
        for i, steps in enumerate(train_traj):       # subsequences of varying lengths
            for _ in range(subsequences if len(steps) > 1 else 0):
                a = rng.randrange(len(steps)); b = rng.randrange(a + 1, len(steps) + 1)
                index += [(i, j) for j in range(a, b)]
        rng.shuffle(index)
        actor.train(); losses = []
        for start in range(0, len(index), batch_size):
            chunk = index[start:start + batch_size]
            ce, coupling, _, _ = _imitation_batch(actor, [train_traj[i][j][0] for i, j in chunk],
                                                  [train_traj[i][j][1] for i, j in chunk], device, coupling_weight)
            loss = (ce + coupling_weight * coupling).mean()
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), .5); optimizer.step()
            losses.append(float(loss.detach()))
        record = dict(epoch=epoch, train_loss=float(np.mean(losses)), **_validate(actor, val_traj, device, coupling_weight, batch_size))
        history.append(record)
        with (output / "history.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        score = record.get("val_ce", record["train_loss"])
        if best is None or score < best:
            best = score
            settings.update(best_epoch=epoch)
            save_checkpoint(output / "model.pt", actor, critic, algorithm, network, config, settings)
        print(f"epoch {epoch + 1}/{epochs}: train_loss={record['train_loss']:.4f} "
              f"val_ce={record.get('val_ce', float('nan')):.4f} val_acc={record.get('val_accuracy', float('nan')):.3f}", flush=True)
    settings.update(history=history)
    (output / "config.json").write_text(json.dumps(dict(simulation=asdict(config), training=settings,
                                                        environment=environment), indent=2))
    return settings


@torch.no_grad()
def _validate(actor, trajectories, device, coupling_weight, batch_size):
    samples = [s for steps in trajectories for s in steps]
    if not samples:
        return {}
    actor.eval(); ce_all, coupling_all, correct_all, roles_all = [], [], [], []
    for start in range(0, len(samples), batch_size):
        chunk = samples[start:start + batch_size]
        ce, coupling, correct, roles = _imitation_batch(actor, [d for d, _ in chunk], [l for _, l in chunk],
                                                        device, coupling_weight)
        ce_all.append(ce.cpu()); coupling_all.append(coupling.cpu()); correct_all.append(correct.cpu()); roles_all.append(roles.cpu())
    ce = torch.cat(ce_all); coupling = torch.cat(coupling_all); correct = torch.cat(correct_all).float(); roles = torch.cat(roles_all)
    out = dict(val_ce=float(ce.mean()), val_coupling=float(coupling.mean()), val_accuracy=float(correct.mean()))
    for r, name in enumerate(("order", "tote", "robot")):
        if (roles == r).any():
            out[f"val_accuracy_{name}"] = float(correct[roles == r].mean())
    return out


def _csv(rows):
    import csv, io
    if not rows:
        return ""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return buffer.getvalue()


def train(config, output, algorithm="MAPPO", instances=None, systems=None, episodes=10, network=None,
          learning_rate=3e-4, gamma=.99, lam=.95, clip=.2, epochs=4, batch_size=64, max_decisions=1_000_000,
          initial_checkpoint=None, device="auto", local_weight=0.1, lambdas=None, entropy_weight=0.01):
    if episodes < 1 or epochs < 1 or max_decisions < 1 or batch_size < 1:
        raise ValueError("Training counts must be positive")
    device = resolve_device(device)
    lambdas = dict(DEFAULT_LAMBDAS if lambdas is None else lambdas)
    instances = expand_instances(instances or [config.instance])
    systems = list(systems or [config.system])
    schedule = [(i, s) for s in systems for i in instances]
    seed_all(config.seed)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    if initial_checkpoint:
        actor, critic, obj = load_checkpoint(initial_checkpoint, device)
        if obj["algorithm"] != algorithm:
            raise ValueError("Initial checkpoint algorithm differs from requested algorithm")
        network = obj["network"]
    else:
        network = network or dict(width=128, heads=12, layers=3)
        actor, critic = make_models(algorithm, **network)
        actor.to(device); critic.to(device)
        warmup, _, _, _ = rollout(replace(config, instance=schedule[0][0], system=schedule[0][1]), actor,
                                  algorithm=algorithm, max_decisions=max_decisions)
        decisions = [s["decision"] for s in warmup]
        actor.fit_standardizers(np.concatenate([d.cont for d in decisions]), np.stack([d.messages for d in decisions]))
        fit_critic(critic, decisions, algorithm)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=learning_rate)
    settings = dict(algorithm=algorithm, instances=instances, systems=systems, episodes=episodes, network=network,
                    learning_rate=learning_rate, gamma=gamma, gae_lambda=lam, clip=clip, ppo_epochs=epochs,
                    batch_size=batch_size, max_decisions=max_decisions, device=device, local_weight=local_weight,
                    lambdas=lambdas, entropy_weight=entropy_weight,
                    initial_checkpoint_sha256=checkpoint_hash(initial_checkpoint) if initial_checkpoint else None)
    (output / "config.json").write_text(json.dumps(dict(simulation=asdict(config), training=settings,
                                                        environment=environment_metadata(device)), indent=2))
    history = []
    for episode in range(episodes):
        instance, system = schedule[episode % len(schedule)]
        run_config = replace(config, instance=instance, system=system, seed=config.seed + episode)
        actor.eval(); critic.eval()
        samples, bootstrap, metrics, _ = rollout(run_config, actor, critic, algorithm, max_decisions=max_decisions,
                                                 local_weight=local_weight, lambdas=lambdas)
        if not samples:
            raise RuntimeError("No transitions collected")
        if algorithm == "IPPO":
            advantages, returns = role_gae(samples, bootstrap, metrics["finished"], gamma, lam, local_weight)
            for role in range(3):
                ids = [i for i, sample in enumerate(samples) if sample["decision"].role == role]
                if ids:
                    values = advantages[ids]
                    advantages[ids] = (values - values.mean()) / (values.std() + 1e-8)
        else:
            advantages, returns = gae([s["reward"] for s in samples], [s["value"] for s in samples], bootstrap,
                                      metrics["finished"], gamma, lam)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = torch.as_tensor(advantages, device=device)
        targets = torch.as_tensor(returns, device=device)
        decisions = [s["decision"] for s in samples]
        actions = torch.as_tensor([s["action"] for s in samples], dtype=torch.long, device=device)
        old_logp = torch.as_tensor([s["log_prob"] for s in samples], dtype=torch.float32, device=device)
        actor.train(); critic.train()
        actor_loss = critic_loss = 0.0
        for _ in range(epochs):
            order = np.random.permutation(len(samples))
            for chunk in np.array_split(order, max(1, math.ceil(len(samples) / batch_size))):
                chunk_t = torch.as_tensor(chunk, dtype=torch.long, device=device)
                batch = collate([decisions[i] for i in chunk], device)
                logits = actor(batch)
                logp = masked_log_softmax(logits, batch["mask"]).gather(1, actions[chunk_t][:, None]).squeeze(1)
                ratio = (logp - old_logp[chunk_t]).exp()
                adv = advantages[chunk_t]
                loss = (-torch.minimum(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
                        - entropy_weight * masked_entropy(logits, batch["mask"]).mean())
                actor_optim.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), .5); actor_optim.step()
                actor_loss = float(loss.detach())
            predicted, _ = critic_values(critic, decisions, algorithm)
            loss = torch.nn.functional.mse_loss(predicted, targets)
            critic_optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), .5); critic_optim.step()
            critic_loss = float(loss.detach())
        record = dict(episode=episode, instance=instance, system=system, seed=run_config.seed,
                      actor_loss=actor_loss, critic_loss=critic_loss,
                      mean_learning_reward=float(np.mean([s["reward"] for s in samples])), **metrics)
        history.append(record)
        with (output / "history.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        save_checkpoint(output / "model.pt", actor, critic, algorithm, network, config, settings)
        print(f"episode {episode + 1}/{episodes} {instance} {system}: moves={metrics['tote_moves']} "
              f"completion={metrics['completion_rate']:.3f}", flush=True)
    return history
