import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from olsf_trs.config import SimulationConfig, configurations, save_instance


def main():
    parser = argparse.ArgumentParser(description="Generate warehouse instances")
    parser.add_argument("--instances", nargs="+", default=["S-1"])
    parser.add_argument("--systems", nargs="+", choices=["multi_tote", "rack_climbing"],
                        default=["multi_tote"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--output", type=Path, default=Path("outputs/data"))
    args = parser.parse_args()
    sizes = configurations()
    instances = list(sizes) if args.instances == ["all"] else args.instances
    if any(name not in sizes for name in instances):
        parser.error("Use S-1 to S-9, L-1 to L-15, or all")
    offset = {"train": 0, "validation": 100000, "test": 200000}[args.split]
    for name in instances:
        for system in args.systems:
            for seed in args.seeds:
                config = SimulationConfig(instance=name, system=system, seed=seed + offset)
                destination = args.output / args.split / system / name / str(config.seed)
                save_instance(config, destination)
                print(destination)


if __name__ == "__main__":
    main()
