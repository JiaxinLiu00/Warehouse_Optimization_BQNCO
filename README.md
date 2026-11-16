# OLSF-TRS

Code for *A scalable learning framework unlocking order fulfillment capacity for
tote-handling robotic systems*.

OLSF-TRS makes order, tote and robot decisions in sequence. It uses a BQ-NCO
Transformer, role attention and a centralized GRU critic. Training combines
expert imitation learning and MAPPO. Baselines include IPPO, C-SGH, R³ and G³.

The simulator supports multi-tote and rack-climbing systems, Poisson order
arrivals, and configurations S-1–S-9 and L-1–L-15. The objective is total tote
movements: retrievals plus returns.

## Environment

`environment.yml` sets up Python 3.11, PyTorch 2.7.1 with CUDA 12.8 and Gurobi 12.0.0:

```bash
conda env create -f environment.yml
conda activate olsf-trs
pip install -e .
```

A CPU installation also works (`pip install -e '.[test,plot]'`). `--device auto` selects CUDA when it
is available. Exact subproblems use Gurobi with a valid licence (`--solver gurobi`) or SciPy/HiGHS
(`--solver scipy`); `--solver auto` picks Gurobi when it is licensed. Run settings and software versions are saved in `manifest.json` or `config.json`.

## Quick check

```bash
python -m pytest -q
python main_train.py evaluate --instance L-1 --seeds 200042 200043 --methods C-SGH R3 G3 \
  --output outputs/check_L1
```

## Experiments

Run pretraining, MAPPO/IPPO training, and evaluation:

```bash
bash scripts/reproduce_paper.sh
```

| Step | Command | Output |
| --- | --- | --- |
| Pretraining | `pretrain --instances all-small --train-instances 1000 --val-instances 300` | `pretrain_*/model.pt`, `history.jsonl`, `demonstrations.csv` |
| MAPPO and IPPO training | `train --instances all-large --initial-checkpoint ...` | `mappo_*/`, `ippo_*/` with `history.jsonl` |
| Small-scale local gaps | `gap --instances all-small --seeds ...` | `gap_*/gap_runs.csv`, `gap_summary.csv` |
| Large-scale results | `evaluate --instances all-large --methods MAPPO IPPO C-SGH R3 G3` | `eval_*/runs.csv`, `summary.csv` |

`runs.csv` stores individual runs;
`summary.csv` stores means, standard deviations and 95% confidence half-widths.

## Settings

Instance sizes and simulation parameters are in `olsf_trs/config.py`. The shared
encoder uses 128 features, three ReZero layers and 12 attention heads.
The centralized critic uses a spatial MLP, GRU and value head.

Candidate grouping uses exact entity attributes within a decision state: equal
order demand, priority and arrival time at the same station; equal tote SKU,
inventory and position; or idle robots at the same position. `--no-quotient`
keeps all entity identities separate.

IPPO computes returns between consecutive decisions of the same role, accumulating
intervening movement rewards with duration-aware discounting.

Local gaps use shared expert states. Order and tote costs include movements and
uncovered-unit penalties; robot costs use pickup trip seconds. Tote evaluation fixes
the expert station and compares retrieval choices. Per-decision costs are saved in
`gap_decisions.csv`. `module_ms` sums the evaluated role's inference time per instance.

Evaluation summaries use the common completed seeds across methods and report
attempted, completed and paired counts.

## Files

| File | Purpose |
| --- | --- |
| `main_train.py` | Command-line entry for pretraining, training and evaluation |
| `olsf_trs/mappo_training.py` | Expert pretraining, recurrent MAPPO and IPPO updates, rollouts and checkpoints |
| `olsf_trs/mappo_models.py` | Centralized GRU critic for MAPPO and separate GRU critics for IPPO |
| `bqnco_framework/encoder.py`, `attention.py` | BQ-NCO encoder and attention layers |
| `bqnco_framework/role_policy.py` | Role policies for order, tote and robot decisions |
| `olsf_trs/config.py` | S-1–S-9 and L-1–L-15 settings and instance generation logic |
| `data/generate_instances.py` | Standalone instance export to `instance.npz` and `metadata.json` |
| `olsf_trs/simulator.py` | Warehouse environment and state transitions |
| `olsf_trs/baselines.py` | Expert solvers, C-SGH, R3 and G3 |
| `olsf_trs/evaluation.py` | Policy evaluation and local gaps |
| `scripts/reproduce_paper.sh` | Full pretraining, training and evaluation workflow |
| `scripts/run_evaluation.py` | Evaluation command entry |
| `scripts/visualize_training.py` | Training curves from saved logs |
| `tests/` | Training, simulation and evaluation checks |

MAPPO and IPPO share the training module; select either with `--algorithm`.
Training generates instances from the settings and seeds. The standalone generator
exports instances for inspection or sharing; its files are not required to start training.

```bash
python main_train.py train --algorithm MAPPO --instances S-1 --output outputs/mappo
python main_train.py train --algorithm IPPO --instances S-1 --output outputs/ippo
python data/generate_instances.py --instances all --seeds 42 --output outputs/data
```

Training curves: `python scripts/visualize_training.py --run-dir outputs/paper/mappo_multi_tote --output curves.png`.

Third-party code and licences are listed in `THIRD_PARTY_NOTICES.md`.
