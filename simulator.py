"""Event-driven goods-to-person warehouse with order, tote and robot decisions."""
from dataclasses import dataclass
import heapq
import itertools
import math
import numpy as np
from .config import generate_instance, instance_fingerprint, storage_layout

ROLES = ("order", "tote", "robot")
KIND_ORDER, KIND_TOTE, KIND_STOP, KIND_ROBOT = 1, 2, 3, 4
CAT_FIELDS = ("kind", "priority", "system")
CAT_VOCAB = (5, 4, 3)
ORDER_FEATURES = ("units", "lines", "overlap_units", "overlap_ratio", "new_lines", "waiting_s",
                  "station_occupancy", "nearest_robot_s", "sku_tote_s", "station_x_m", "free_slots")
TOTE_FEATURES = ("units_served", "inventory", "station_need", "emptied", "travel_s", "route_increment_s",
                 "cargo", "x_m", "y_m", "z_m", "station_occupancy")
ROBOT_FEATURES = ("route_s", "first_leg_s", "x_m", "y_m", "z_m", "station_distance_m", "cargo",
                  "capacity", "free_robots", "station_occupancy", "station_x_m")
ROLE_FEATURES = (ORDER_FEATURES, TOTE_FEATURES, ROBOT_FEATURES)
CONT_DIM = len(ORDER_FEATURES) + 1      # role attributes + log class size
MSG_DIM = 8
GLOBAL_DIM = 3 * MSG_DIM + 6
NO_TOTE_SECONDS = 3600.0


@dataclass
class Decision:
    role: int
    actions: list          # representative action of each class
    members: list          # all member actions of each class
    cont: np.ndarray       # [classes, CONT_DIM] class-mean continuous attributes and log class size
    cat: np.ndarray        # [classes, len(CAT_FIELDS)] categorical attributes
    messages: np.ndarray   # [3, MSG_DIM] role observations used for communication
    global_state: np.ndarray
    raw_candidates: int
    classes: int
    quotiented: bool

    def __post_init__(self):
        self.lookup = {a: i for i, group in enumerate(self.members) for a in group}

    @property
    def counts(self):
        return np.array([len(group) for group in self.members])

    def index_of(self, action):
        return self.lookup.get(tuple(int(a) for a in action))


def _bins(values, width):
    values = np.asarray(values, dtype=float)
    return np.floor(values / width).astype(np.int64) if width > 0 else np.round(values, 6)


def _signature(row):
    ids = np.flatnonzero(row)
    return tuple(zip(ids.tolist(), row[ids].tolist()))


class Warehouse:
    def __init__(self, config, data=None, policy_roles=(0, 1, 2)):
        self.config = config
        self.policy_roles = frozenset(policy_roles)
        self.data = generate_instance(config) if data is None else {k: np.array(v, copy=True) for k, v in data.items()}
        self.fingerprint = instance_fingerprint(self.data)
        self.layout = storage_layout(config)
        n = self.n = config.size
        self.capacity = config.robot_capacity
        self.system_code = 1 if config.system == "multi_tote" else 2
        self.remaining = self.data["demand"].copy()
        self.inventory = self.data["inventory"].copy()
        self.assignments = np.full(n["orders"], -1, dtype=int)
        self.released = np.zeros(n["orders"], dtype=bool)
        self.complete = np.zeros(n["orders"], dtype=bool)
        self.tote_free = np.ones(n["totes"], dtype=bool)
        self.robot_free = np.ones(n["robots"], dtype=bool)
        self.robot_positions = np.array(self.data["robot_positions"], dtype=float, copy=True)
        self.station_busy = np.zeros(n["workstations"], dtype=bool)
        self.time = 0.0
        self.events = []
        self.serial = 0
        self.retrievals = 0
        self.returns = 0
        self.travel_seconds = 0.0
        self.decisions = 0
        self.cargo = []
        self.target_station = None
        self.robot_stage = False
        self.last_info = None
        self.bq_raw = np.zeros(3, dtype=np.int64)
        self.bq_classes = np.zeros(3, dtype=np.int64)
        self._decision = None
        self._routes = {}
        self._paths = {}
        self._order_signature = [_signature(row) for row in self.data["demand"]]
        self._tote_station = self._travel_matrix(self.data["tote_positions"], self.data["station_positions"])
        for order, t in enumerate(self.data["arrivals"]):
            self._schedule(t, "arrival", order)

    def travel(self, start, end):
        start = np.asarray(start, dtype=float); end = np.asarray(end, dtype=float)
        return float(np.linalg.norm(start[:2] - end[:2]) / self.config.horizontal_speed_m_s
                     + abs(start[2] - end[2]) / self.config.vertical_speed_m_s)

    def _travel_to(self, points, end):
        points = np.asarray(points, dtype=float); end = np.asarray(end, dtype=float)
        return (np.linalg.norm(points[:, :2] - end[:2], axis=1) / self.config.horizontal_speed_m_s
                + np.abs(points[:, 2] - end[2]) / self.config.vertical_speed_m_s)

    def _travel_matrix(self, a, b):
        a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
        horizontal = np.linalg.norm(a[:, None, :2] - b[None, :, :2], axis=2)
        return horizontal / self.config.horizontal_speed_m_s + np.abs(a[:, None, 2] - b[None, :, 2]) / self.config.vertical_speed_m_s

    def _cargo_paths(self, cargo, station):
        """Shortest pickup path from each possible first tote through the cargo to the station."""
        key = (tuple(cargo), station)
        if key not in self._paths:
            pos = self.data["tote_positions"]; target = self.data["station_positions"][station]
            best = {}
            for perm in itertools.permutations(cargo):
                total = sum(self.travel(pos[a], pos[b]) for a, b in zip(perm[:-1], perm[1:]))
                total += self.travel(pos[perm[-1]], target)
                if perm[0] not in best or total < best[perm[0]][0] - 1e-12:
                    best[perm[0]] = (total, list(perm))
            self._paths[key] = best
        return self._paths[key]

    def _robot_routes(self, robots):
        """Exact route (robot start, optimal pickup sequence, workstation) for each robot."""
        best = self._cargo_paths(self.cargo, self.target_station)
        firsts = list(best)
        legs = self._travel_matrix(self.robot_positions[robots], self.data["tote_positions"][firsts])
        totals = legs + np.array([best[f][0] for f in firsts])[None, :]
        choice = totals.argmin(axis=1)
        return {int(r): (float(totals[i, choice[i]]), best[firsts[choice[i]]][1], float(legs[i, choice[i]]))
                for i, r in enumerate(robots)}

    def _insertion_cost(self, totes, station):
        if not self.cargo:
            return self._tote_station[totes, station]
        tail, sequence = min(self._cargo_paths(self.cargo, station).values(), key=lambda v: v[0])
        nodes = [self.data["tote_positions"][t] for t in sequence] + [self.data["station_positions"][station]]
        points = self.data["tote_positions"][totes]
        costs = [self._travel_to(points, nodes[0])]
        for a, b in zip(nodes[:-1], nodes[1:]):
            costs.append(self._travel_to(points, a) + self._travel_to(points, b) - self.travel(a, b))
        return np.min(costs, axis=0)

    def _nearest_robot_time(self, station):
        free = np.flatnonzero(self.robot_free)
        robots = free if len(free) else np.arange(self.n["robots"])
        return float(self._travel_to(self.robot_positions[robots], self.data["station_positions"][station]).min())

    def _schedule(self, t, kind, payload):
        self.serial += 1
        heapq.heappush(self.events, (float(t), self.serial, kind, payload))

    @property
    def moves(self):
        return self.retrievals + self.returns

    @property
    def done(self):
        return bool(self.complete.all() and not self.events and not self.cargo)

    def active(self, station):
        return np.flatnonzero((self.assignments == station) & ~self.complete)

    def needs(self, station, cargo=False):
        need = self.remaining[self.active(station)].sum(axis=0)
        if cargo and self.target_station == station:
            for tote in self.cargo:
                sku = self.data["tote_sku"][tote]
                need[sku] = max(0, need[sku] - self.inventory[tote])
        return need

    def _event(self):
        if not self.events:
            raise RuntimeError("No feasible action or future event; inspect inventory and task state")
        self.time, _, kind, payload = heapq.heappop(self.events)
        if kind == "arrival":
            self.released[payload] = True
        elif kind == "delivery":
            robot, sequence, station = payload
            units = 0
            for tote in sequence:
                sku = self.data["tote_sku"][tote]
                for order in sorted(self.active(station), key=lambda o: (-self.data["priority"][o], self.data["arrivals"][o], o)):
                    qty = min(self.inventory[tote], self.remaining[order, sku])
                    self.remaining[order, sku] -= qty
                    self.inventory[tote] -= qty
                    units += int(qty)
            service = units * self.config.pick_seconds_per_unit + len(sequence) * self.config.handling_seconds_per_tote
            self._schedule(self.time + service, "picked", payload)
        elif kind == "picked":
            robot, sequence, station = payload
            active = self.active(station)
            self.complete[active] = self.remaining[active].sum(axis=1) == 0
            self.station_busy[station] = False
            returning = [t for t in sequence if self.inventory[t] > 0]
            start = self.data["station_positions"][station]
            best = None
            for perm in itertools.permutations(returning):
                total, pos, times = 0.0, start, []
                for tote in perm:
                    total += self.travel(pos, self.data["tote_positions"][tote]); pos = self.data["tote_positions"][tote]
                    times.append(total)
                if best is None or total < best[0] - 1e-12:
                    best = (total, perm, times, pos)
            total, perm, times, end = best if best is not None else (0.0, (), [], start)
            for tote, elapsed in zip(perm, times):
                self._schedule(self.time + elapsed, "return_tote", tote)
            self.travel_seconds += total
            self._schedule(self.time + total, "robot_free", (robot, np.array(end, dtype=float)))
        elif kind == "return_tote":
            self.returns += 1
            self.tote_free[payload] = True
        elif kind == "robot_free":
            robot, pos = payload
            self.robot_positions[robot] = pos
            self.robot_free[robot] = True

    def _messages(self):
        n = self.n
        left = self.remaining.sum() / max(1, self.data["demand"].sum())
        pending = self.released & (self.assignments < 0)
        waiting = float((self.time - self.data["arrivals"][pending]).mean()) if pending.any() else 0.0
        needed = sum(int((self.needs(s) > 0).sum()) for s in range(n["workstations"]))
        m = np.zeros((3, MSG_DIM), dtype=np.float32)
        m[0] = [self.released.mean(), self.complete.mean(), left, ((self.assignments >= 0) & ~self.complete).mean(),
                self.time / 600, self.station_busy.mean(), pending.mean(), waiting / 600]
        m[1] = [self.tote_free.mean(), self.inventory.sum() / max(1, self.data["inventory"].sum()),
                len(self.cargo) / self.capacity, self.retrievals / n["totes"], self.returns / n["totes"], left,
                self.moves / n["orders"], needed / n["skus"]]
        m[2] = [self.robot_free.mean(), self.capacity / 4,
                self.robot_positions[:, 0].mean() / max(1.0, self.layout["width_m"]),
                self.robot_positions[:, 1].mean() / max(1.0, self.layout["depth_m"]),
                self.robot_positions[:, 2].mean() / max(1.0, self.config.rack_levels * self.config.level_height_m),
                self.station_busy.mean(), self.travel_seconds / max(1.0, self.time * n["robots"]), self.time / 600]
        return m

    def _order_entries(self):
        pending = np.flatnonzero(self.released & (self.assignments < 0))
        if not len(pending):
            return []
        slots = self.config.slots_per_station
        demand = self.remaining[pending]
        units = demand.sum(axis=1); lines = (demand > 0).sum(axis=1)
        available = np.flatnonzero(self.tote_free & (self.inventory > 0))
        sku_time = np.full((self.n["skus"], self.n["workstations"]), NO_TOTE_SECONDS)
        np.minimum.at(sku_time, self.data["tote_sku"][available], self._tote_station[available])
        priority = self.data["priority"]; arrivals = self.data["arrivals"]
        entries = []
        for s in range(self.n["workstations"]):
            if self.station_busy[s]:
                continue
            active = self.active(s)
            free_slots = slots - len(active)
            if free_slots <= 0:
                continue
            needs = self.needs(s)
            overlap = np.minimum(demand, needs[None, :]).sum(axis=1)
            new_lines = ((demand > 0) & (needs[None, :] == 0)).sum(axis=1)
            nearest = self._nearest_robot_time(s)
            tote_time = (demand > 0) @ sku_time[:, s] / np.maximum(lines, 1)
            waiting = self.time - arrivals[pending]
            occupancy = len(active) / slots
            station_signature = _signature(needs)
            wait_bins = _bins(waiting, self.config.bq_time_bin_s)
            tote_bins = _bins(tote_time, self.config.bq_time_bin_s)
            nearest_bin = _bins([nearest], self.config.bq_time_bin_s)[0]
            station_x = self.data["station_positions"][s, 0]
            for i, o in enumerate(pending):
                cont = (units[i], lines[i], overlap[i], overlap[i] / max(1, units[i]), new_lines[i], waiting[i],
                        occupancy, nearest, tote_time[i], station_x, free_slots)
                key = (self._order_signature[o], station_signature, len(active), wait_bins[i], nearest_bin, tote_bins[i])
                rank = overlap[i] + 0.01 * priority[o] - 1e-4 * nearest
                entries.append(((int(o), s), cont, (KIND_ORDER, int(priority[o]), self.system_code), key, float(rank)))
        return entries

    def _tote_entries(self):
        entries = []
        slots = self.config.slots_per_station
        stations = [self.target_station] if self.cargo else range(self.n["workstations"])
        sku_of = self.data["tote_sku"]
        for s in stations:
            if self.station_busy[s]:
                continue
            need = self.needs(s, cargo=True)
            idx = np.flatnonzero(self.tote_free & (self.inventory > 0) & (need[sku_of] > 0))
            if not len(idx):
                continue
            sku = sku_of[idx]; inv = self.inventory[idx]; req = need[sku]
            qty = np.minimum(inv, req); emptied = (inv <= req).astype(float)
            travel = self._tote_station[idx, s]
            increment = self._insertion_cost(idx, s)
            pos = self.data["tote_positions"][idx]
            occupancy = len(self.active(s)) / slots
            cont = np.column_stack([qty, inv, req, emptied, travel, increment, np.full(len(idx), len(self.cargo)),
                                    pos[:, 0], pos[:, 1], pos[:, 2], np.full(len(idx), occupancy)])
            travel_bins = _bins(travel, self.config.bq_time_bin_s)
            increment_bins = _bins(increment, self.config.bq_time_bin_s)
            for j, t in enumerate(idx):
                key = (int(sku[j]), int(inv[j]), int(qty[j]), int(emptied[j]), s, travel_bins[j], increment_bins[j])
                entries.append(((int(t), s), cont[j], (KIND_TOTE, 0, self.system_code), key,
                                float(qty[j] - 0.001 * increment[j])))
        return entries

    def _robot_entries(self):
        robots = np.flatnonzero(self.robot_free)
        self._routes = self._robot_routes(robots)
        station = self.target_station
        target = self.data["station_positions"][station]
        occupancy = len(self.active(station)) / self.config.slots_per_station
        entries = []
        for r in robots:
            route, _, leg = self._routes[int(r)]
            pos = self.robot_positions[r]
            distance = float(np.linalg.norm(pos[:2] - target[:2]))
            cont = (route, leg, pos[0], pos[1], pos[2], distance, len(self.cargo), self.capacity,
                    len(robots), occupancy, target[0])
            key = (_bins([route], self.config.bq_time_bin_s)[0], _bins([distance], self.config.bq_distance_bin_m)[0])
            entries.append(((int(r),), cont, (KIND_ROBOT, 0, self.system_code), key, -route))
        return entries

    def _equivalence_key(self, role, action):
        if role == 0:
            order, station = action
            return (station, tuple(self.remaining[order]), int(self.data["priority"][order]),
                    float(self.data["arrivals"][order]))
        if role == 1:
            tote, station = action
            if tote < 0:
                return ("stop", station)
            return (station, int(self.data["tote_sku"][tote]), int(self.inventory[tote]),
                    tuple(self.data["tote_positions"][tote]))
        return tuple(self.robot_positions[action[0]])

    def _make(self, role, entries):
        raw = len(entries)
        entries.sort(key=lambda e: (-e[4], e[0]))
        policy = role in self.policy_roles
        quotiented = policy and self.config.quotient
        if quotiented:
            groups = {}
            for e in entries:
                groups.setdefault(self._equivalence_key(role, e[0]), []).append(e)
            classes = list(groups.values())
        else:
            classes = [[e] for e in entries]
        n_classes = len(classes)
        if policy:                  # the policy scores every class; the stop action comes last
            classes = ([c for c in classes if c[0][2][0] != KIND_STOP]
                       + [c for c in classes if c[0][2][0] == KIND_STOP])
        self.bq_raw[role] += raw
        self.bq_classes[role] += n_classes
        cont = np.zeros((len(classes), CONT_DIM), dtype=np.float32)
        cat = np.zeros((len(classes), len(CAT_FIELDS)), dtype=np.int64)
        for i, group in enumerate(classes):
            cont[i, :-1] = group[0][1] if len(group) == 1 else np.mean([e[1] for e in group], axis=0)
            cont[i, -1] = math.log(len(group))
            cat[i] = group[0][2]
        members = [[e[0] for e in group] for group in classes]
        messages = self._messages()
        n = self.n
        global_state = np.concatenate([messages.reshape(-1), np.array(
            [role / 2, self.time / 600, self.moves / n["orders"], n["orders"] / 350, n["totes"] / 5000,
             float(self.config.system == "rack_climbing")], dtype=np.float32)])
        self._decision = Decision(role, [m[0] for m in members], members, cont, cat, messages, global_state,
                                  raw, n_classes, quotiented)
        return self._decision

    def decision(self):
        if self._decision is not None:
            return self._decision
        while not self.done:
            while self.events and self.events[0][0] <= self.time:
                self._event()
            if self.done:
                return None
            if self.robot_stage:
                if not self.robot_free.any():
                    self._event()
                    continue
                return self._make(2, self._robot_entries())
            if not self.cargo:
                entries = self._order_entries()
                if entries:
                    return self._make(0, entries)
            if self.robot_free.any():
                entries = self._tote_entries()
                if self.cargo and (len(self.cargo) >= self.capacity or not entries):
                    self.robot_stage = True
                    continue
                if self.cargo:
                    entries.append(((-1, self.target_station), np.zeros(CONT_DIM - 1), (KIND_STOP, 0, self.system_code),
                                    ("stop",), -1e9))
                if entries:
                    return self._make(1, entries)
            self._event()
        return None

    def feasible(self, role, action):
        if role == 0:
            o, s = action
            return (0 <= o < self.n["orders"] and 0 <= s < self.n["workstations"] and not self.cargo
                    and self.released[o] and self.assignments[o] < 0 and not self.station_busy[s]
                    and len(self.active(s)) < self.config.slots_per_station)
        if role == 1:
            t, s = action
            if t < 0:
                return bool(self.cargo) and s == self.target_station
            if not (0 <= t < self.n["totes"] and 0 <= s < self.n["workstations"]):
                return False
            if self.cargo and (s != self.target_station or len(self.cargo) >= self.capacity):
                return False
            return (self.tote_free[t] and self.inventory[t] > 0 and not self.station_busy[s]
                    and self.needs(s, cargo=True)[self.data["tote_sku"][t]] > 0)
        r = action[0]
        return self.robot_stage and 0 <= r < self.n["robots"] and bool(self.robot_free[r])

    def _local_terms(self, role, action):
        """Intrinsic objective f_i and impacts on the other roles for R_i = f_i + sum_j lambda_ij Impact_ij."""
        if role == 0:
            o, s = action
            demand = self.remaining[o]; needs = self.needs(s)
            units = max(1, int(demand.sum())); lines = max(1, int((demand > 0).sum()))
            overlap = float(np.minimum(demand, needs).sum())
            new_lines = float(((demand > 0) & (needs == 0)).sum())
            return dict(role=0, f=overlap / units, impacts={1: -new_lines / lines, 2: -self._nearest_robot_time(s) / 60})
        if role == 1:
            t, s = action
            if t < 0:
                return dict(role=1, f=0.0, impacts={0: 0.0, 2: 0.0})
            need = self.needs(s, cargo=True)
            qty = float(min(self.inventory[t], need[self.data["tote_sku"][t]]))
            increment = float(self._insertion_cost(np.array([t]), s)[0])
            return dict(role=1, f=qty / max(1, int(self.inventory[t])),
                        impacts={0: qty / max(1.0, float(need.sum())), 2: -increment / 60})
        route = self._routes[action[0]][0]
        active = len(self.active(self.target_station))
        return dict(role=2, f=-route / 60,
                    impacts={0: -route * active / (60 * self.config.slots_per_station),
                             1: -route * len(self.cargo) / (60 * self.capacity)})

    def _apply(self, role, action):
        before = self.moves
        info = self._local_terms(role, action)
        if role == 0:
            order, station = action
            self.assignments[order] = station
        elif role == 1:
            tote, station = action
            if tote < 0:
                self.robot_stage = True
            else:
                self.cargo.append(tote)
                self.target_station = station
                self.tote_free[tote] = False
        else:
            robot = action[0]
            route, sequence, _ = self._routes[robot]
            self.robot_free[robot] = False
            self.station_busy[self.target_station] = True
            self.travel_seconds += route
            self.retrievals += len(self.cargo)
            self._schedule(self.time + route, "delivery", (robot, list(sequence), self.target_station))
            self.cargo = []
            self.target_station = None
            self.robot_stage = False
        self.decisions += 1
        self._decision = None
        self.last_info = info
        nxt = self.decision()
        return nxt, float(before - self.moves), self.done

    def step(self, index):
        """Apply the representative action of class ``index`` of the current decision."""
        d = self.decision()
        if d is None or not 0 <= index < len(d.actions):
            raise ValueError("Infeasible action")
        return self._apply(d.role, d.actions[index])

    def step_action(self, action):
        """Apply a specific feasible action (used by exact experts and rule-based baselines)."""
        d = self.decision()
        action = tuple(int(a) for a in action)
        if d is None or not self.feasible(d.role, action):
            raise ValueError("Infeasible action")
        if d.role == 2 and action[0] not in self._routes:
            self._routes = self._robot_routes(np.flatnonzero(self.robot_free))
        return self._apply(d.role, action)

    def metrics(self):
        raw = self.bq_raw.sum(); classes = self.bq_classes.sum()
        out = dict(tote_moves=self.moves, retrievals=self.retrievals, returns=self.returns,
                   completed_orders=int(self.complete.sum()), orders=self.n["orders"],
                   completion_rate=float(self.complete.mean()), simulation_seconds=self.time,
                   robot_travel_seconds=self.travel_seconds, decisions=self.decisions,
                   remaining_units=int(self.remaining.sum()), finished=self.done,
                   instance_fingerprint=self.fingerprint,
                   bq_candidate_reduction=float(1 - classes / raw) if raw else 0.0)
        for i, name in enumerate(ROLES):
            out[f"bq_raw_{name}"] = int(self.bq_raw[i]); out[f"bq_classes_{name}"] = int(self.bq_classes[i])
        return out
