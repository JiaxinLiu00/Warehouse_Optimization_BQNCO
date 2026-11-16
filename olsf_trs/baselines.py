"""Rule-based baselines (R3, G3, C-SGH) and exact local subproblem solvers."""
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment, milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix
from .simulator import KIND_STOP

HEURISTICS = ("R3", "G3", "C-SGH")
UNCOVERED_UNIT_PENALTY = 3.0   # exceeds the two movements needed to retrieve and return one tote


def resolve_solver(solver="auto"):
    """Return "gurobi" when requested or available with a working licence, otherwise "scipy"."""
    if solver in ("scipy", "gurobi"):
        return solver
    if solver != "auto":
        raise ValueError("Unknown solver " + str(solver))
    try:
        import gurobipy as gp
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0); env.start()
            with gp.Model(env=env) as model:
                x = model.addVar(vtype=gp.GRB.BINARY); model.setObjective(x); model.optimize()
        return "gurobi"
    except Exception:
        return "scipy"


def solver_version(solver):
    if solver == "gurobi":
        import gurobipy as gp
        return "Gurobi " + ".".join(map(str, gp.gurobi.version()))
    import scipy
    return "SciPy/HiGHS " + scipy.__version__


class MILP:
    """Minimal sparse MILP builder with Gurobi and SciPy/HiGHS back ends."""

    def __init__(self):
        self.cost, self.lower, self.upper, self.integer = [], [], [], []
        self.rows, self.cols, self.vals, self.lo, self.hi = [], [], [], [], []

    def var(self, cost=0.0, lower=0.0, upper=1.0, integer=True):
        self.cost.append(float(cost)); self.lower.append(float(lower)); self.upper.append(float(upper))
        self.integer.append(bool(integer))
        return len(self.cost) - 1

    def constraint(self, terms, lo=-np.inf, hi=np.inf):
        row = len(self.lo)
        for col, val in terms:
            self.rows.append(row); self.cols.append(col); self.vals.append(float(val))
        self.lo.append(float(lo)); self.hi.append(float(hi))

    def solve(self, solver="scipy"):
        n = len(self.cost)
        a = csr_matrix((self.vals, (self.rows, self.cols)), shape=(len(self.lo), n))
        lo = np.asarray(self.lo); hi = np.asarray(self.hi)
        if solver == "gurobi":
            import gurobipy as gp
            with gp.Env(empty=True) as env:
                env.setParam("OutputFlag", 0); env.start()
                with gp.Model(env=env) as model:
                    vtype = ["B" if i and l == 0 and u == 1 else ("I" if i else "C")
                             for i, l, u in zip(self.integer, self.lower, self.upper)]
                    x = model.addMVar(n, lb=self.lower, ub=self.upper, vtype=vtype)
                    finite_lo = np.isfinite(lo); finite_hi = np.isfinite(hi)
                    if finite_lo.any():
                        model.addConstr(a[finite_lo] @ x >= lo[finite_lo])
                    if finite_hi.any():
                        model.addConstr(a[finite_hi] @ x <= hi[finite_hi])
                    model.setObjective(np.asarray(self.cost) @ x, gp.GRB.MINIMIZE)
                    model.optimize()
                    if model.Status != gp.GRB.OPTIMAL:
                        raise RuntimeError("Gurobi did not establish optimality (status %d)" % model.Status)
                    return np.asarray(x.X)
        if solver != "scipy":
            raise ValueError("Unknown solver " + str(solver))
        constraints = LinearConstraint(a, lo, hi) if len(self.lo) else ()
        result = milp(np.asarray(self.cost), integrality=np.asarray(self.integer, dtype=int),
                      bounds=Bounds(self.lower, self.upper), constraints=constraints)
        if not result.success:
            raise RuntimeError("HiGHS did not establish optimality: " + str(result.message))
        return result.x


def tote_groups(env, totes):
    """Totes with equal SKU and inventory are interchangeable in the movement objective."""
    groups = {}
    for t in totes:
        groups.setdefault((int(env.data["tote_sku"][t]), int(env.inventory[t])), []).append(int(t))
    return groups


def movement_cover(groups, demand, solver="scipy", forced_group=None):
    """Exact minimum tote movements (retrievals plus returns) to cover SKU demand.

    groups: {(sku, inventory): [totes]}; demand: {sku: units}. Returns {(sku, inventory): (retrieved, emptied, usage)}.
    """
    model = MILP(); variables = {}; by_sku = {}
    for (sku, inv), members in groups.items():
        if demand.get(sku, 0) <= 0 or inv <= 0:
            continue
        count = len(members)
        y = model.var(2.0, 0, count); e = model.var(-1.0, 0, count); z = model.var(0.0, 0, inv * count, integer=False)
        model.constraint([(z, 1), (y, -inv)], hi=0)
        model.constraint([(z, 1), (e, -inv)], lo=0)
        model.constraint([(e, 1), (y, -1)], hi=0)
        variables[(sku, inv)] = (y, e, z); by_sku.setdefault(sku, []).append(z)
    for sku, units in demand.items():
        if units > 0:
            u = model.var(UNCOVERED_UNIT_PENALTY, 0, np.inf, integer=False)
            model.constraint([(z, 1) for z in by_sku.get(sku, [])] + [(u, 1)], lo=units, hi=units)
    if forced_group is not None:
        y, _, z = variables[forced_group]
        model.constraint([(y, 1)], lo=1)
        model.constraint([(z, 1)], lo=1)
    if not variables:
        return {}
    x = model.solve(solver)
    return {g: (int(round(x[y])), int(round(x[e])), float(x[z])) for g, (y, e, z) in variables.items()}


def minimum_tote_cover(inventory, sku, demand, solver="scipy"):
    """Exact minimum number of totes covering a station's demand."""
    inventory = np.asarray(inventory); sku = np.asarray(sku); demand = np.asarray(demand)
    if demand.sum() == 0:
        return np.zeros(len(inventory), dtype=bool)
    matrix = np.zeros((len(demand), len(inventory)))
    matrix[sku, np.arange(len(inventory))] = inventory
    if np.any(matrix.sum(1) < demand):
        raise ValueError("Inventory cannot cover station demand")
    model = MILP()
    xs = [model.var(1.0) for _ in range(len(inventory))]
    for k in range(len(demand)):
        if demand[k] > 0:
            model.constraint([(xs[t], matrix[k, t]) for t in range(len(inventory)) if matrix[k, t] > 0], lo=demand[k])
    return model.solve(solver) > 0.5


def order_assignment_plan(env, solver="scipy", forced_pair=None, return_cost=False):
    """Exact local order assignment: fill workstation slots, then minimize required tote movements."""
    pending = np.flatnonzero(env.released & (env.assignments < 0))
    stations = [s for s in range(env.n["workstations"]) if not env.station_busy[s]
                and len(env.active(s)) < env.config.slots_per_station]
    if not len(pending) or not stations:
        return []
    available = np.flatnonzero(env.tote_free & (env.inventory > 0))
    groups = tote_groups(env, available)
    demand = env.remaining
    skus = set(np.flatnonzero(demand[pending].sum(0)).tolist())
    needs = {s: env.needs(s) for s in stations}
    max_units = int(demand[pending].sum(1).max())
    big = UNCOVERED_UNIT_PENALTY * max_units + 1
    model = MILP(); x = {}; usage = {}
    for o in pending:
        for s in stations:
            x[(int(o), s)] = model.var(-big)
    group_vars = {g: [] for g in groups}
    for s in stations:
        relevant = skus | set(np.flatnonzero(needs[s]).tolist())
        by_sku = {}
        for (sku, inv), members in groups.items():
            if sku not in relevant:
                continue
            y = model.var(2.0, 0, len(members)); e = model.var(-1.0, 0, len(members))
            z = model.var(0.0, 0, inv * len(members), integer=False)
            model.constraint([(z, 1), (y, -inv)], hi=0)
            model.constraint([(z, 1), (e, -inv)], lo=0)
            model.constraint([(e, 1), (y, -1)], hi=0)
            group_vars[(sku, inv)].append(y); by_sku.setdefault(sku, []).append(z)
        for sku in relevant:
            u = model.var(UNCOVERED_UNIT_PENALTY, 0, np.inf, integer=False)
            terms = [(z, 1) for z in by_sku.get(sku, [])] + [(u, 1)]
            terms += [(x[(int(o), s)], -demand[o, sku]) for o in pending if demand[o, sku] > 0]
            model.constraint(terms, lo=needs[s][sku], hi=needs[s][sku])
    for g, ys in group_vars.items():
        if ys:
            model.constraint([(y, 1) for y in ys], hi=len(groups[g]))
    for o in pending:
        model.constraint([(x[(int(o), s)], 1) for s in stations], hi=1)
    for s in stations:
        free = env.config.slots_per_station - len(env.active(s))
        model.constraint([(x[(int(o), s)], 1) for o in pending], hi=free)
    if forced_pair is not None:
        model.constraint([(x[tuple(forced_pair)], 1)], lo=1, hi=1)
    solution = model.solve(solver)
    chosen = [pair for pair, v in x.items() if solution[v] > 0.5]
    if return_cost:
        return float(np.dot(model.cost, solution) + big * len(chosen))
    priority = env.data["priority"]; arrivals = env.data["arrivals"]
    return sorted(chosen, key=lambda p: (-priority[p[0]], arrivals[p[0]], p))


def tote_trip_plan(env, solver="scipy"):
    """Exact local tote matching: the station and totes with the fewest movements per covered unit."""
    stations = [env.target_station] if env.cargo else [s for s in range(env.n["workstations"]) if not env.station_busy[s]]
    best = None
    for s in stations:
        need = env.needs(s, cargo=True)
        if need.sum() == 0:
            continue
        available = np.flatnonzero(env.tote_free & (env.inventory > 0) & (need[env.data["tote_sku"]] > 0))
        if not len(available):
            continue
        groups = tote_groups(env, available)
        plan = movement_cover(groups, {int(k): int(need[k]) for k in np.flatnonzero(need)}, solver)
        moves = sum(2 * y - e for y, e, _ in plan.values())
        covered = sum(z for _, _, z in plan.values())
        if moves <= 0 or covered <= 0:
            continue
        totes = []
        for g, (y, e, z) in plan.items():
            members = sorted(groups[g], key=lambda t: (env._tote_station[t, s], t))[:y]
            per_tote = z / max(1, y)
            totes += [(per_tote, env._tote_station[t, s], t) for t in members]
        totes = [t for _, _, t in sorted(totes, key=lambda v: (-v[0], v[1], v[2]))]
        score = (covered / moves, covered, -s)
        if best is None or score > best[0]:
            best = (score, s, totes)
    if best is None:
        return None, []
    return best[1], best[2]


def robot_assignment(env, solver="scipy"):
    """Exact robot assignment and pickup sequencing minimizing the trip completion time (MTZ model)."""
    robots = np.flatnonzero(env.robot_free)
    cargo = list(env.cargo); m = len(cargo)
    pos = env.data["tote_positions"]; station = env.data["station_positions"][env.target_station]
    model = MILP(); first = {}; arc = {}; last = {}; order = {}
    for r in robots:
        for j in range(m):
            first[(int(r), j)] = model.var(env.travel(env.robot_positions[r], pos[cargo[j]]))
    for i in range(m):
        for j in range(m):
            if i != j:
                arc[(i, j)] = model.var(env.travel(pos[cargo[i]], pos[cargo[j]]))
        last[i] = model.var(env.travel(pos[cargo[i]], station))
        order[i] = model.var(0.0, 1, m, integer=False)
    model.constraint([(v, 1) for v in first.values()], lo=1, hi=1)
    for j in range(m):
        incoming = [(first[(int(r), j)], 1) for r in robots] + [(arc[(i, j)], 1) for i in range(m) if i != j]
        model.constraint(incoming, lo=1, hi=1)
        model.constraint([(arc[(j, k)], 1) for k in range(m) if k != j] + [(last[j], 1)], lo=1, hi=1)
    model.constraint([(v, 1) for v in last.values()], lo=1, hi=1)
    for (i, j), v in arc.items():
        model.constraint([(order[i], 1), (order[j], -1), (v, m)], hi=m - 1)
    x = model.solve(solver)
    robot = max(first, key=lambda key: x[first[key]])[0]
    return (int(robot),)


def expert_action(env, decision, solver="scipy"):
    """Exact local decision for the current role, reused within one decision epoch when still valid."""
    cache = getattr(env, "_expert_cache", None)
    valid = cache is not None and cache["time"] == env.time and cache["next"] == env.decisions and cache["role"] == decision.role
    if decision.role == 0:
        if not valid:
            cache = dict(role=0, queue=order_assignment_plan(env, solver))
        queue = [p for p in cache["queue"] if env.feasible(0, p)]
        if not queue:
            queue = [p for p in order_assignment_plan(env, solver) if env.feasible(0, p)]
        if not queue:
            raise RuntimeError("Exact order assignment produced no feasible pair")
        action, cache["queue"] = queue[0], queue[1:]
    elif decision.role == 1:
        if not valid:
            station, totes = tote_trip_plan(env, solver)
            cache = dict(role=1, station=station, queue=totes)
        room = env.capacity - len(env.cargo)
        queue = [t for t in cache["queue"] if env.feasible(1, (t, cache["station"]))]
        if room > 0 and queue:
            action, cache["queue"] = (queue[0], cache["station"]), queue[1:]
        elif env.cargo:
            action = (-1, env.target_station)
        else:
            raise RuntimeError("Exact tote matching produced no feasible tote")
    else:
        action = robot_assignment(env, solver)
        cache = dict(role=2)
    cache.update(time=env.time, next=env.decisions + 1)
    env._expert_cache = cache
    return action


def _first(order):
    return min(order)[1]


def choose(env, decision, method="C-SGH"):
    """Rule-based baselines operate on the complete, unquotiented candidate set."""
    if method not in HEURISTICS:
        raise ValueError("Unknown baseline " + str(method))
    cont, cat, actions = decision.cont, decision.cat, decision.actions
    real = [i for i in range(len(actions)) if cat[i, 0] != KIND_STOP]
    if decision.role == 0:
        if method == "R3":   # robot-first: workstation closest to an available robot
            return actions[_first(((cont[i, 7], -cont[i, 5], actions[i]), i) for i in range(len(actions)))]
        return actions[_first(((-cont[i, 2], -cat[i, 1], -cont[i, 5], actions[i]), i) for i in range(len(actions)))]
    if decision.role == 2:
        planned = getattr(env, "collaborative_robot", None)
        if method == "C-SGH" and planned is not None and (planned,) in actions:
            return (planned,)
        return actions[_first(((cont[i, 0], actions[i]), i) for i in range(len(actions)))]
    if not real:
        return actions[0]
    if method == "R3":       # robot travel efficiency
        return actions[_first(((cont[i, 5], actions[i]), i) for i in real)]
    if method == "G3":       # group-based greedy gathering: most units per tote
        return actions[_first(((-cont[i, 0], cont[i, 4], actions[i]), i) for i in real)]
    return _collaborative_tote(env, decision, real)


def _collaborative_tote(env, decision, real, alpha=1.0, beta=0.01):
    """C-SGH: maximum-weight bipartite matching between free robots and candidate totes."""
    robots = np.flatnonzero(env.robot_free)
    sku_of = env.data["tote_sku"]
    utility = np.empty((len(robots), len(real)))
    for ci, i in enumerate(real):
        tote, station = decision.actions[i]
        active = env.active(station)
        batch = float((env.remaining[active, sku_of[tote]] > 0).sum())   # active orders served by the tote
        travel = env._travel_to(env.robot_positions[robots], env.data["tote_positions"][tote]) + env._tote_station[tote, station]
        utility[:, ci] = alpha * batch - beta * travel
    rows, cols = linear_sum_assignment(-utility)
    best = max(zip(rows, cols), key=lambda rc: (utility[rc], -rc[1]))
    env.collaborative_robot = int(robots[best[0]])
    return decision.actions[real[best[1]]]


def local_decision_costs(env, action, solver="scipy"):
    """Optimal local objective and its optimum conditioned on the selected action."""
    role = env.decision().role
    if role == 0:
        ref = order_assignment_plan(env, solver, return_cost=True)
        selected = order_assignment_plan(env, solver, forced_pair=action, return_cost=True)
        return ref, selected, "movements_plus_uncovered_unit_penalty"
    if role == 2:
        best = robot_assignment(env, solver)[0]
        return env._routes[best][0], env._routes[action[0]][0], "trip_seconds"
    tote, station = action
    if tote < 0:
        raise ValueError("Local tote-cover evaluation requires a retrieval action")
    need = env.needs(station, cargo=True)
    available = np.flatnonzero(env.tote_free & (env.inventory > 0) & (need[env.data["tote_sku"]] > 0))
    groups = tote_groups(env, available)
    demand = {int(k): int(need[k]) for k in np.flatnonzero(need)}
    def objective(plan):
        return (sum(2 * y - e for y, e, _ in plan.values())
                + UNCOVERED_UNIT_PENALTY * (sum(demand.values()) - sum(z for _, _, z in plan.values())))
    group = (int(env.data["tote_sku"][tote]), int(env.inventory[tote]))
    return (objective(movement_cover(groups, demand, solver)),
            objective(movement_cover(groups, demand, solver, forced_group=group)),
            "movements_plus_uncovered_unit_penalty")
