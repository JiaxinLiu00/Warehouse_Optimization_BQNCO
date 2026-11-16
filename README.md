# Data generation

Generate warehouse instances with a fixed seed. Run from the repository root:

```bash
python data/generate_instances.py --instances S-1 --seeds 42 --output outputs/data

python data/generate_instances.py --instances all --systems multi_tote rack_climbing \
  --seeds 42 43 44 --split test --output outputs/data
```

Each instance folder contains `instance.npz` and `metadata.json`.

| Array | Contents |
| --- | --- |
| `demand` | Order quantities by SKU |
| `tote_sku` | SKU index for each tote (one SKU per tote, several totes per SKU) |
| `inventory` | Initial quantity in each tote |
| `tote_positions` | Storage cell of each tote (metres; x longitudinal, y transverse, z level height) |
| `station_positions` | Workstation coordinates (metres) |
| `robot_positions` | Initial robot positions in the aisles (metres) |
| `arrivals` | Order arrival times (seconds), Poisson process with rate N_o/10 orders per minute |
| `priority` | Order priorities (1 to 3) |

The storage area consists of rack modules arranged along longitudinal and transverse
aisles. Each module has `cells_per_module` cells on `rack_levels` levels; the number of
modules is chosen so that the totes fill `storage_fill` of the storage locations. Both
architectures use the same layout and tote positions for a given seed. Multi-tote robots
carry up to four totes and reach levels with an onboard lift; rack-climbing robots carry
one tote and climb the rack. The layout summary is saved in `metadata.json`.

Seed offsets for train, validation and test are 0, 100000 and 200000. Instance sizes and
generation settings are defined in `olsf_trs/config.py`.

```python
import numpy as np

instance = np.load("outputs/data/train/multi_tote/S-1/42/instance.npz")
demand = instance["demand"]
```
