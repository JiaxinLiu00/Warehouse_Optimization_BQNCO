import argparse
from dataclasses import replace
import json
from pathlib import Path
import torch
from .config import SimulationConfig, SPLIT_OFFSETS, configurations, expand_instances, save_instance
from .mappo_training import train, pretrain, DEFAULT_LAMBDAS
from .evaluation import evaluate, gap_evaluation, METHODS


def _lambdas(values):
    lambdas = dict(DEFAULT_LAMBDAS)
    for item in values or []:
        pair, value = item.split("=")
        lambdas[pair.strip()] = float(value)
    return lambdas


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m olsf_trs")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["generate", "pretrain", "train", "evaluate", "gap"]:
        p = sub.add_parser(name)
        p.add_argument("--instance", default="S-1", choices=list(configurations()))
        p.add_argument("--system", choices=["multi_tote", "rack_climbing"], default="multi_tote")
        p.add_argument("--seed", type=int, default=200042 if name in ("evaluate", "gap") else 42)
        p.add_argument("--output", required=True, type=Path)
        p.add_argument("--slots", type=int, default=4)
        p.add_argument("--static-arrivals", action="store_true")
        p.add_argument("--no-quotient", action="store_true", help="keep individual entity actions")
        p.add_argument("--bq-time-bin", type=float, default=0.0, help="exact entity grouping uses 0")
        p.add_argument("--bq-distance-bin", type=float, default=0.0, help="exact entity grouping uses 0")
        p.add_argument("--threads", type=int, default=1)
        if name != "generate":
            p.add_argument("--max-decisions", type=int, default=1_000_000)
            p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto",
                           help="auto selects CUDA when available")
        if name in ("pretrain", "train"):
            p.add_argument("--algorithm", choices=["MAPPO", "IPPO"], default="MAPPO")
            p.add_argument("--width", type=int, default=128)
            p.add_argument("--heads", type=int, default=12)
            p.add_argument("--layers", type=int, default=3)
            p.add_argument("--learning-rate", type=float, default=3e-4)
            p.add_argument("--batch-size", type=int, default=64)
            p.add_argument("--instances", nargs="+", help="instances or all-small / all-large")
            p.add_argument("--systems", nargs="+", choices=["multi_tote", "rack_climbing"])
        if name == "pretrain":
            p.add_argument("--train-instances", type=int, default=1000)
            p.add_argument("--val-instances", type=int, default=300)
            p.add_argument("--epochs", type=int, default=30)
            p.add_argument("--coupling-weight", type=float, default=.1)
            p.add_argument("--subsequences", type=int, default=2)
            p.add_argument("--solver", choices=["auto", "gurobi", "scipy"], default="auto")
        if name == "train":
            p.add_argument("--episodes", type=int, default=10)
            p.add_argument("--epochs", type=int, default=4)
            p.add_argument("--initial-checkpoint", type=Path)
            p.add_argument("--gamma", type=float, default=.99)
            p.add_argument("--gae-lambda", type=float, default=.95)
            p.add_argument("--clip", type=float, default=.2)
            p.add_argument("--entropy-weight", type=float, default=.01)
            p.add_argument("--local-weight", type=float, default=.1, help="weight of the shaped local reward R_i")
            p.add_argument("--lambda", dest="lambdas", action="append", metavar="I,J=VALUE",
                           help="inter-agent weight lambda_ij (roles 0 order, 1 tote, 2 robot)")
        if name in ("evaluate", "gap"):
            p.add_argument("--instances", nargs="+", help="instances or all-small / all-large")
            p.add_argument("--systems", nargs="+", choices=["multi_tote", "rack_climbing"])
            p.add_argument("--seeds", nargs="+", type=int)
            p.add_argument("--checkpoint", type=Path, required=name == "gap")
            p.add_argument("--solver", choices=["auto", "gurobi", "scipy"], default="auto")
        if name == "evaluate":
            p.add_argument("--methods", nargs="+", choices=list(METHODS), default=["R3", "G3", "C-SGH"])
            p.add_argument("--ippo-checkpoint", type=Path)
            p.add_argument("--save-traces", action="store_true")
        if name == "generate":
            p.add_argument("--split", choices=["train", "validation", "test"], default="train")
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    config = SimulationConfig(instance=args.instance, system=args.system, seed=args.seed, slots_per_station=args.slots,
                              stochastic_arrivals=not args.static_arrivals,
                              quotient=not args.no_quotient, bq_time_bin_s=args.bq_time_bin,
                              bq_distance_bin_m=args.bq_distance_bin)
    if args.command == "generate":
        if args.output.exists():
            raise FileExistsError(args.output)
        result = save_instance(replace(config, seed=config.seed + SPLIT_OFFSETS[args.split]), args.output)
    elif args.command == "pretrain":
        result = pretrain(config, args.output, args.instances or ["all-small"], args.systems or [args.system],
                          args.train_instances, args.val_instances, args.epochs,
                          dict(width=args.width, heads=args.heads, layers=args.layers), args.coupling_weight,
                          args.solver, args.device, args.batch_size, args.learning_rate, args.subsequences,
                          args.algorithm, args.max_decisions)
        result = {k: v for k, v in result.items() if k != "history"}
    elif args.command == "train":
        history = train(config, args.output, args.algorithm, args.instances or [args.instance], args.systems or [args.system],
                        args.episodes, dict(width=args.width, heads=args.heads, layers=args.layers), args.learning_rate,
                        args.gamma, args.gae_lambda, args.clip, args.epochs, args.batch_size, args.max_decisions,
                        args.initial_checkpoint, args.device, args.local_weight, _lambdas(args.lambdas),
                        args.entropy_weight)
        result = dict(episodes=len(history), last=history[-1]["tote_moves"])
    elif args.command == "evaluate":
        instances = expand_instances(args.instances or [args.instance])
        rows = evaluate(config, args.output, instances, args.systems or [args.system], args.seeds or [args.seed],
                        args.methods, args.checkpoint, args.ippo_checkpoint, args.max_decisions, args.device,
                        args.solver, args.save_traces)
        result = dict(runs=len(rows), output=str(args.output))
    else:
        instances = expand_instances(args.instances or ["all-small"])
        rows = gap_evaluation(config, args.output, instances, args.systems or [args.system], args.seeds or [args.seed],
                              args.checkpoint, args.solver, args.device, args.max_decisions)
        result = dict(runs=len(rows), output=str(args.output))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
