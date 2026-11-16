"""Instance configurations, storage layout and synthetic instance generation."""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import math
import numpy as np

INSTANCE_SIZES = {
    'S-1': {'skus': 20, 'orders': 10, 'robots': 3, 'workstations': 3, 'totes': 100},
    'S-2': {'skus': 25, 'orders': 12, 'robots': 4, 'workstations': 3, 'totes': 125},
    'S-3': {'skus': 30, 'orders': 14, 'robots': 4, 'workstations': 4, 'totes': 150},
    'S-4': {'skus': 35, 'orders': 15, 'robots': 5, 'workstations': 4, 'totes': 175},
    'S-5': {'skus': 40, 'orders': 15, 'robots': 5, 'workstations': 5, 'totes': 200},
    'S-6': {'skus': 45, 'orders': 16, 'robots': 5, 'workstations': 5, 'totes': 225},
    'S-7': {'skus': 50, 'orders': 18, 'robots': 6, 'workstations': 5, 'totes': 250},
    'S-8': {'skus': 55, 'orders': 18, 'robots': 6, 'workstations': 6, 'totes': 275},
    'S-9': {'skus': 60, 'orders': 20, 'robots': 6, 'workstations': 6, 'totes': 300},
    'L-1': {'skus': 60, 'orders': 50, 'robots': 10, 'workstations': 5, 'totes': 500},
    'L-2': {'skus': 80, 'orders': 60, 'robots': 15, 'workstations': 5, 'totes': 650},
    'L-3': {'skus': 100, 'orders': 70, 'robots': 20, 'workstations': 5, 'totes': 800},
    'L-4': {'skus': 120, 'orders': 80, 'robots': 25, 'workstations': 10, 'totes': 1000},
    'L-5': {'skus': 140, 'orders': 90, 'robots': 30, 'workstations': 10, 'totes': 1200},
    'L-6': {'skus': 160, 'orders': 100, 'robots': 35, 'workstations': 10, 'totes': 1400},
    'L-7': {'skus': 180, 'orders': 110, 'robots': 40, 'workstations': 15, 'totes': 1600},
    'L-8': {'skus': 200, 'orders': 120, 'robots': 45, 'workstations': 15, 'totes': 1800},
    'L-9': {'skus': 220, 'orders': 130, 'robots': 50, 'workstations': 15, 'totes': 2000},
    'L-10': {'skus': 250, 'orders': 150, 'robots': 55, 'workstations': 15, 'totes': 2200},
    'L-11': {'skus': 300, 'orders': 180, 'robots': 60, 'workstations': 20, 'totes': 2500},
    'L-12': {'skus': 350, 'orders': 210, 'robots': 60, 'workstations': 20, 'totes': 3000},
    'L-13': {'skus': 400, 'orders': 250, 'robots': 65, 'workstations': 20, 'totes': 3500},
    'L-14': {'skus': 500, 'orders': 300, 'robots': 65, 'workstations': 20, 'totes': 4000},
    'L-15': {'skus': 600, 'orders': 350, 'robots': 70, 'workstations': 25, 'totes': 5000},
}
SMALL_INSTANCES = [f"S-{i}" for i in range(1, 10)]
LARGE_INSTANCES = [f"L-{i}" for i in range(1, 16)]
SPLIT_OFFSETS = {"train": 0, "validation": 100000, "test": 200000}
SYSTEMS = ("multi_tote", "rack_climbing")


def configurations():
    return {name: size.copy() for name, size in INSTANCE_SIZES.items()}


def expand_instances(names):
    """Expand the aliases ``all``, ``all-small`` and ``all-large``."""
    out = []
    for name in names:
        if name == "all":
            out += list(INSTANCE_SIZES)
        elif name == "all-small":
            out += SMALL_INSTANCES
        elif name == "all-large":
            out += LARGE_INSTANCES
        elif name in INSTANCE_SIZES:
            out.append(name)
        else:
            raise ValueError("Unknown instance: " + name)
    return out


@dataclass
class SimulationConfig:
    instance: str = "S-1"
    system: str = "multi_tote"
    seed: int = 42
    slots_per_station: int = 4
    carrying_capacity: int = 4
    horizontal_speed_m_s: float = 1.5
    climbing_speed_m_s: float = 0.5      # rack-climbing robots travel vertically on the rack
    lift_speed_m_s: float = 1.0          # multi-tote robots reach levels with an onboard lift
    pick_seconds_per_unit: float = 2.0
    handling_seconds_per_tote: float = 2.0
    order_skus_max: int = 4
    quantity_max: int = 3
    arrival_window_seconds: float = 600.0   # Poisson rate N_o/10 orders per minute
    stochastic_arrivals: bool = True
    cells_per_module: int = 8
    cell_width_m: float = 1.0
    module_depth_m: float = 1.0
    aisle_width_m: float = 2.0
    rack_levels: int = 8
    level_height_m: float = 1.0
    storage_fill: float = 0.85
    quotient: bool = True
    bq_time_bin_s: float = 0.0
    bq_distance_bin_m: float = 0.0

    def __post_init__(self):
        if self.instance not in INSTANCE_SIZES:
            raise ValueError("Unknown instance: " + self.instance)
        if self.system not in SYSTEMS:
            raise ValueError("Unknown system: " + self.system)
        for key in ("slots_per_station", "carrying_capacity", "order_skus_max",
                    "quantity_max", "cells_per_module", "rack_levels"):
            if getattr(self, key) < 1:
                raise ValueError(key + " must be positive")
        for key in ("horizontal_speed_m_s", "climbing_speed_m_s", "lift_speed_m_s", "arrival_window_seconds",
                    "cell_width_m", "module_depth_m", "aisle_width_m", "level_height_m"):
            if getattr(self, key) <= 0:
                raise ValueError(key + " must be positive")
        if not 0 < self.storage_fill <= 1:
            raise ValueError("storage_fill must be in (0, 1]")
        if self.pick_seconds_per_unit < 0 or self.handling_seconds_per_tote < 0:
            raise ValueError("Service times must be nonnegative")
        if self.bq_time_bin_s != 0 or self.bq_distance_bin_m != 0:
            raise ValueError("Exact entity grouping requires zero bin widths")

    @property
    def size(self):
        return configurations()[self.instance]

    @property
    def robot_capacity(self):
        return self.carrying_capacity if self.system == "multi_tote" else 1

    @property
    def vertical_speed_m_s(self):
        return self.lift_speed_m_s if self.system == "multi_tote" else self.climbing_speed_m_s


def storage_layout(config):
    """Storage cells of the rack modules, shared by both robot architectures."""
    needed = math.ceil(config.size["totes"] / config.storage_fill)
    per_module = config.cells_per_module * config.rack_levels
    modules = math.ceil(needed / per_module)
    module_length = config.cells_per_module * config.cell_width_m
    pitch_x = module_length + config.aisle_width_m          # transverse aisles between modules
    pitch_y = config.module_depth_m + config.aisle_width_m  # longitudinal aisles between rows
    per_row = max(1, min(modules, round(math.sqrt(modules * pitch_y / pitch_x))))
    rows = math.ceil(modules / per_row)
    cells = []
    for module in range(modules):
        row, column = divmod(module, per_row)
        for cell in range(config.cells_per_module):
            x = column * pitch_x + (cell + 0.5) * config.cell_width_m
            y = row * pitch_y + 0.5 * config.module_depth_m
            for level in range(config.rack_levels):
                cells.append((x, y, level * config.level_height_m))
    return dict(cells=np.asarray(cells, dtype=np.float64), modules=modules, modules_per_row=per_row, rows=rows,
                width_m=per_row * pitch_x - config.aisle_width_m, depth_m=rows * pitch_y - config.aisle_width_m)


def generate_instance(config):
    """Demand, inventory, storage positions, robot positions and Poisson order arrivals."""
    rng = np.random.default_rng(config.seed)
    n = config.size
    demand = np.zeros((n["orders"], n["skus"]), dtype=np.int64)
    for row in demand:
        k = rng.integers(1, min(config.order_skus_max, n["skus"]) + 1)
        ids = rng.choice(n["skus"], k, replace=False)
        row[ids] = rng.integers(1, config.quantity_max + 1, k)
    sku = np.arange(n["totes"], dtype=np.int64) % n["skus"]
    rng.shuffle(sku)
    counts = np.bincount(sku, minlength=n["skus"])
    stock = np.ceil(demand.sum(0) / counts).astype(np.int64) + config.quantity_max
    inventory = stock[sku]
    layout = storage_layout(config)
    cells = layout["cells"]
    tote_positions = cells[rng.choice(len(cells), n["totes"], replace=False)]
    floor = cells[cells[:, 2] == 0]
    robot_cells = floor[rng.choice(len(floor), n["robots"], replace=len(floor) < n["robots"])]
    aisle_offset = 0.5 * (config.module_depth_m + config.aisle_width_m)
    robot_positions = robot_cells + np.array([0.0, aisle_offset, 0.0])
    stations = np.column_stack([np.linspace(0, layout["width_m"], n["workstations"]),
                                np.full(n["workstations"], -5.0), np.zeros(n["workstations"])])
    arrivals = (np.cumsum(rng.exponential(config.arrival_window_seconds / n["orders"], n["orders"]))
                if config.stochastic_arrivals else np.zeros(n["orders"]))
    return dict(demand=demand, tote_sku=sku, inventory=inventory, tote_positions=tote_positions,
                station_positions=stations, robot_positions=robot_positions, arrivals=arrivals,
                priority=rng.integers(1, 4, n["orders"]))


def instance_fingerprint(data):
    h = hashlib.sha256()
    for key in sorted(data):
        value = np.ascontiguousarray(data[key])
        h.update(key.encode()); h.update(str(value.shape).encode())
        h.update(value.dtype.str.encode()); h.update(value.tobytes())
    return h.hexdigest()


def save_instance(config, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    data = generate_instance(config)
    np.savez_compressed(destination / "instance.npz", **data)
    layout = storage_layout(config)
    metadata = dict(config=asdict(config), size=config.size, fingerprint=instance_fingerprint(data),
                    arrival_rate_orders_per_minute=config.size["orders"] * 60 / config.arrival_window_seconds,
                    layout={k: v for k, v in layout.items() if k != "cells"},
                    storage_locations=int(len(layout["cells"])),
                    units=dict(arrivals="seconds", positions="metres", demand="item units"))
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
