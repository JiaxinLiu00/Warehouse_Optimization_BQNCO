import argparse
import json
from pathlib import Path


def load_history(run_dir):
    path = Path(run_dir) / "history.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Empty training log: {path}")
    required = {"episode", "tote_moves", "completion_rate", "actor_loss", "critic_loss"}
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"Missing log fields: {sorted(missing)}")
    return rows


def plot_history(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    episodes = [row["episode"] + 1 for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True)
    panels = [("tote_moves", "Tote movements", 1),
              ("completion_rate", "Order completion (%)", 100),
              ("actor_loss", "Actor loss", 1),
              ("critic_loss", "Critic loss", 1)]
    for ax, (key, label, scale) in zip(axes.flat, panels):
        ax.plot(episodes, [row[key] * scale for row in rows], marker="o", markersize=3)
        ax.set(xlabel="Episode", ylabel=label)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=.2)
    axes[0, 1].set_ylim(0, 105)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description="Plot training history")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(plot_history(load_history(args.run_dir), args.output))


if __name__ == "__main__":
    main()
