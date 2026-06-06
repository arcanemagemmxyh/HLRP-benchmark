# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _import_common():
    import importlib.util
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent / 'evaluator' / 'hlrp_instance.py',
        Path.cwd() / 'src' / 'evaluator' / 'hlrp_instance.py',
    ]
    for f in candidates:
        if f.exists():
            spec = importlib.util.spec_from_file_location('_hlrp_common_runtime', f)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            return mod.HLRPInstance, mod.load_instance
    for f in [here.parent / 'hlrp_instance_v2.py', Path.cwd() / 'hlrp_instance_v2.py']:
        if f.exists():
            spec = importlib.util.spec_from_file_location('_hlrp_common_runtime_v2', f)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            return mod.HLRPInstance, mod.load_instance
    import hlrp_instance as mod  # type: ignore
    return mod.HLRPInstance, mod.load_instance


HLRPInstance, load_instance = _import_common()


def _assign_cost(inst: HLRPInstance, customer: int, hub: int) -> float:
    if customer == hub:
        return 0.0
    if hasattr(inst, 'assignment_cost_of'):
        return float(inst.assignment_cost_of(customer, hub))
    w = float(getattr(inst, 'assignment_weight', 0.0))
    d = float(inst.distance[customer][hub])
    tp = list(getattr(inst, 'node_throughput_normalized', [1.0] * inst.n))
    scale = float(tp[customer]) if customer < len(tp) else 1.0
    return w * scale * d


def _hub_loads(inst: HLRPInstance, assignment: Dict[int, int], open_hubs: List[int]) -> Dict[int, float]:
    if hasattr(inst, 'compute_hub_loads'):
        return {int(k): float(v) for k, v in inst.compute_hub_loads(assignment, open_hubs).items()}
    loads = {int(h): 0.0 for h in open_hubs}
    outbound = list(getattr(inst, 'outbound_flow', [0.0] * inst.n))
    inbound = list(getattr(inst, 'inbound_flow', [0.0] * inst.n))
    for i, h in assignment.items():
        if i in loads:
            continue
        loads[h] = loads.get(h, 0.0) + float(outbound[i]) + float(inbound[i])
    return loads


def _node_load(inst: HLRPInstance, node: int) -> float:
    outbound = list(getattr(inst, 'outbound_flow', [0.0] * inst.n))
    inbound = list(getattr(inst, 'inbound_flow', [0.0] * inst.n))
    return float(outbound[node]) + float(inbound[node])


def _repair_capacity(inst: HLRPInstance, ch: Chromosome, max_open: int, max_rounds: int = 100) -> None:
    ch.assignment_hub = _normalize_assignment(inst, ch.assignment_hub)
    _rebuild_perm(ch, inst)
    for _ in range(max_rounds):
        open_hubs = sorted(set(ch.assignment_hub.values()))
        loads = _hub_loads(inst, ch.assignment_hub, open_hubs)
        overloaded = [h for h in open_hubs if loads.get(h, 0.0) - inst.hub_capacity[h] > 1e-9]
        if not overloaded:
            return
        changed = False
        # Try direct reassignment to hubs with slack first
        for h in sorted(overloaded, key=lambda x: loads.get(x, 0.0) - inst.hub_capacity[x], reverse=True):
            customers_h = [c for c, hh in ch.assignment_hub.items() if hh == h and c != h]
            if not customers_h:
                continue
            slack_hubs = [g for g in open_hubs if g != h and inst.hub_capacity[g] - loads.get(g, 0.0) > 1e-9]
            best_move = None
            best_key = None
            for c in customers_h:
                load_c = _node_load(inst, c)
                for g in slack_hubs:
                    if inst.hub_capacity[g] - loads.get(g, 0.0) + 1e-9 < load_c:
                        continue
                    delta = _assign_cost(inst, c, g) - _assign_cost(inst, c, h)
                    key = (delta, -load_c)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_move = (c, g)
            if best_move is not None:
                c, g = best_move
                ch.assignment_hub[c] = g
                changed = True
                break
            # If direct reassignment fails, open a new hub from a customer of this overloaded hub
            if len(open_hubs) < max_open:
                customers_h = sorted(customers_h, key=lambda c: (_node_load(inst, c), -inst.hub_fixed_cost[c], -_assign_cost(inst, c, h)), reverse=True)
                if customers_h:
                    new_h = customers_h[0]
                    # New hub becomes opened, so remove it from customer assignments.
                    ch.assignment_hub.pop(new_h, None)
                    current_open = sorted(set(ch.assignment_hub.values()) | {new_h})
                    # Move some customers from h to new_h greedily until h is no longer overloaded or new_h fills.
                    loads2 = _hub_loads(inst, ch.assignment_hub, current_open)
                    overload = max(0.0, loads2.get(h, 0.0) - inst.hub_capacity[h])
                    cap_new = inst.hub_capacity[new_h] - loads2.get(new_h, 0.0)
                    movable = [c for c in customers_h[1:] if c in ch.assignment_hub]
                    movable.sort(key=lambda c: (_assign_cost(inst, c, h) - _assign_cost(inst, c, new_h), _node_load(inst, c)))
                    for c in movable:
                        if overload <= 1e-9 or cap_new <= 1e-9:
                            break
                        load_c = _node_load(inst, c)
                        if load_c <= cap_new + 1e-9:
                            ch.assignment_hub[c] = new_h
                            overload -= load_c
                            cap_new -= load_c
                    changed = True
                    break
        if not changed:
            return
        ch.assignment_hub = _normalize_assignment(inst, ch.assignment_hub)
        _rebuild_perm(ch, inst)


@dataclass
class Chromosome:
    assignment_hub: Dict[int, int]           # customer node -> hub node
    permutation: List[int]                   # order of customer nodes only
    fitness: Optional[float] = None
    objective: Optional[float] = None
    penalty: Optional[float] = None
    open_hubs: Optional[List[int]] = None
    routes: Optional[Dict[int, List[List[int]]]] = None
    feasible: Optional[bool] = None
    violations: Optional[List[str]] = None
    costs: Optional[Dict[str, float]] = None

    def copy(self) -> 'Chromosome':
        return Chromosome(
            assignment_hub=dict(self.assignment_hub),
            permutation=list(self.permutation),
            fitness=self.fitness,
            objective=self.objective,
            penalty=self.penalty,
            open_hubs=None if self.open_hubs is None else list(self.open_hubs),
            routes=None if self.routes is None else {h: [list(r) for r in rs] for h, rs in self.routes.items()},
            feasible=self.feasible,
            violations=None if self.violations is None else list(self.violations),
            costs=None if self.costs is None else dict(self.costs),
        )


def _customers(inst: HLRPInstance, open_hubs: List[int]) -> List[int]:
    open_set = set(open_hubs)
    return [i for i in range(inst.n) if i not in open_set]


def _rebuild_perm(ch: Chromosome, inst: HLRPInstance) -> None:
    ch.assignment_hub = _normalize_assignment(inst, ch.assignment_hub)
    open_hubs = sorted(set(ch.assignment_hub.values()))
    custs = set(_customers(inst, open_hubs))
    ordered: List[int] = []
    seen = set()
    for c in ch.permutation:
        if c in custs and c not in seen:
            ordered.append(c)
            seen.add(c)
    for c in sorted(custs):
        if c not in seen:
            ordered.append(c)
    groups: Dict[int, List[int]] = {h: [] for h in open_hubs}
    for c in ordered:
        h = ch.assignment_hub[c]
        groups.setdefault(h, []).append(c)
    rebuilt: List[int] = []
    for h in open_hubs:
        rebuilt.extend(groups.get(h, []))
    ch.permutation = rebuilt


def decode_routes(inst: HLRPInstance, ch: Chromosome) -> Dict[int, List[List[int]]]:
    open_hubs = sorted(set(ch.assignment_hub.values()))
    buckets: Dict[int, List[int]] = {h: [] for h in open_hubs}
    for c in ch.permutation:
        h = ch.assignment_hub.get(c)
        if h is not None and c != h:
            buckets.setdefault(h, []).append(c)
    routes: Dict[int, List[List[int]]] = {h: [] for h in open_hubs}
    for h in open_hubs:
        custs = buckets.get(h, [])
        for s in range(0, len(custs), inst.q):
            part = [c for c in custs[s:s + inst.q] if c != h]
            if part:
                routes[h].append(part)
    return routes


def _solution_from_chromosome(inst: HLRPInstance, ch: Chromosome) -> Dict[str, object]:
    open_hubs = sorted(set(ch.assignment_hub.values()))
    custs = _customers(inst, open_hubs)
    assignment = {str(c): int(ch.assignment_hub[c]) for c in custs}
    routes = {str(h): decode_routes(inst, ch).get(h, []) for h in open_hubs}
    return {
        'open_hubs': open_hubs,
        'assignment': assignment,
        'routes': routes,
    }


def _manual_objective(inst: HLRPInstance, solution: Dict[str, object]) -> Tuple[float, Dict[str, float], Dict[int, float]]:
    open_hubs = sorted(int(x) for x in solution['open_hubs'])
    assignment = {int(k): int(v) for k, v in solution['assignment'].items()}  # type: ignore[index]
    routes = {int(h): [[int(x) for x in rt] for rt in rts] for h, rts in solution['routes'].items()}  # type: ignore[index]
    hub = float(sum(inst.hub_fixed_cost[h] for h in open_hubs))
    route = float(sum(inst.route_cost(h, rt) for h, rs in routes.items() for rt in rs))
    assign = float(sum(_assign_cost(inst, i, h) for i, h in assignment.items() if i != h))
    inter = float(inst.compute_inter_hub_flow_cost(assignment))
    vehicle = float(inst.vehicle_fixed_cost_param * sum(len(rs) for rs in routes.values()))
    loads = _hub_loads(inst, assignment, open_hubs)
    total = hub + route + assign + inter + vehicle
    return total, {
        'hub': hub,
        'route': route,
        'assignment': assign,
        'interhub': inter,
        'vehicle': vehicle,
    }, loads


def evaluate(inst: HLRPInstance, ch: Chromosome, delta: float = 10000.0) -> float:
    _rebuild_perm(ch, inst)
    sol = _solution_from_chromosome(inst, ch)
    obj, costs, loads = _manual_objective(inst, sol)
    penalty = 0.0
    open_hubs = sol['open_hubs']  # type: ignore[assignment]
    for h in open_hubs:
        cap = inst.hub_capacity[h]
        load = loads.get(h, 0.0)
        penalty += delta * max(0.0, load - cap)
    res = inst.evaluate_solution(sol)
    if not res.feasible:
        penalty += delta * max(1, len(res.violations))
    ch.objective = float(obj)
    ch.penalty = float(penalty)
    ch.fitness = float(obj + penalty)
    ch.open_hubs = list(sol['open_hubs'])  # type: ignore[arg-type]
    ch.routes = {int(h): [list(rt) for rt in rs] for h, rs in sol['routes'].items()}  # type: ignore[index]
    ch.feasible = bool(res.feasible)
    ch.violations = list(res.violations)
    ch.costs = dict(costs)
    return ch.fitness


def _nearest_hub(inst: HLRPInstance, c: int, hubs: List[int]) -> int:
    return min(hubs, key=lambda h: _assign_cost(inst, c, h))


def random_chromosome(inst: HLRPInstance, rng: random.Random, max_open: int) -> Chromosome:
    k = rng.randint(1, max_open)
    open_hubs = sorted(rng.sample(list(range(inst.n)), k=k))
    customers = _customers(inst, open_hubs)
    assignment: Dict[int, int] = {}
    for c in customers:
        if rng.random() < 0.75:
            assignment[c] = _nearest_hub(inst, c, open_hubs)
        else:
            assignment[c] = rng.choice(open_hubs)
    perm = list(customers)
    rng.shuffle(perm)
    ch = Chromosome(assignment_hub=_normalize_assignment(inst, assignment), permutation=perm)
    _repair_capacity(inst, ch, max_open=max_open)
    evaluate(inst, ch)
    return ch


def rank_selection(pop: List[Chromosome], rng: random.Random) -> Chromosome:
    srt = sorted(pop, key=lambda c: c.fitness)
    n = len(srt)
    probs = [2 * (n - rank) / (n * (n + 1)) for rank in range(n)]
    r = rng.random()
    acc = 0.0
    for ch, p in zip(srt, probs):
        acc += p
        if r <= acc:
            return ch
    return srt[-1]


def crossover_assignment(a: Chromosome, b: Chromosome, rng: random.Random) -> Dict[int, int]:
    keys = sorted(set(a.assignment_hub) | set(b.assignment_hub))
    if not keys:
        return {}
    cut = rng.randint(1, max(1, len(keys) - 1)) if len(keys) > 1 else 1
    out: Dict[int, int] = {}
    for idx, c in enumerate(keys):
        if idx < cut:
            out[c] = a.assignment_hub.get(c, b.assignment_hub.get(c))
        else:
            out[c] = b.assignment_hub.get(c, a.assignment_hub.get(c))
    return out


def crossover_perm(a: List[int], b: List[int], rng: random.Random) -> List[int]:
    if len(a) <= 1:
        return list(a)
    cut = rng.randint(1, len(a) - 1)
    prefix = a[:cut]
    used = set(prefix)
    return prefix + [x for x in b if x not in used]


def _normalize_assignment(inst: HLRPInstance, assignment: Dict[int, int]) -> Dict[int, int]:
    open_hubs = sorted(set(int(h) for h in assignment.values()))
    if not open_hubs:
        open_hubs = [0]
    normalized: Dict[int, int] = {}
    customers = _customers(inst, open_hubs)
    for c in customers:
        h = assignment.get(c)
        if h is None or h not in open_hubs or c == h:
            normalized[c] = _nearest_hub(inst, c, open_hubs)
        else:
            normalized[c] = int(h)
    return normalized


def mutate_assignment(inst: HLRPInstance, ch: Chromosome, pm: float, rng: random.Random, max_open: int) -> None:
    if rng.random() > pm:
        return
    open_hubs = sorted(set(ch.assignment_hub.values()))
    move = rng.random()
    if move < 0.33 and len(open_hubs) > 1:
        close_h = rng.choice(open_hubs)
        remain = [h for h in open_hubs if h != close_h]
        for c, h in list(ch.assignment_hub.items()):
            if h == close_h:
                ch.assignment_hub[c] = _nearest_hub(inst, c, remain)
    elif move < 0.66 and len(open_hubs) < max_open:
        closed = [i for i in range(inst.n) if i not in open_hubs]
        if closed:
            new_h = rng.choice(closed)
            for c in list(ch.assignment_hub.keys()):
                if rng.random() < 0.25:
                    new_cost = _assign_cost(inst, c, new_h)
                    old_cost = _assign_cost(inst, c, ch.assignment_hub[c])
                    if new_cost <= old_cost * 1.2:
                        ch.assignment_hub[c] = new_h
    else:
        customers = list(ch.assignment_hub.keys())
        if customers:
            for c in rng.sample(customers, k=max(1, len(customers) // 6)):
                ch.assignment_hub[c] = rng.choice(open_hubs)
    ch.assignment_hub = _normalize_assignment(inst, ch.assignment_hub)
    _rebuild_perm(ch, inst)


def mutate_perm(ch: Chromosome, pm: float, rng: random.Random) -> None:
    if rng.random() > pm or len(ch.permutation) <= 1:
        return
    i, j = sorted(rng.sample(range(len(ch.permutation)), 2))
    node = ch.permutation.pop(i)
    ch.permutation.insert(j if j <= len(ch.permutation) else len(ch.permutation), node)


def make_offspring(inst: HLRPInstance, p1: Chromosome, p2: Chromosome, rng: random.Random, max_open: int, pc: float, pm1: float, pm2: float, delta: float) -> Chromosome:
    if rng.random() <= pc:
        assignment = crossover_assignment(p1, p2, rng)
        temp = Chromosome(assignment_hub=assignment, permutation=crossover_perm(p1.permutation, p2.permutation, rng))
    else:
        temp = p1.copy()
    temp.assignment_hub = _normalize_assignment(inst, temp.assignment_hub)
    _rebuild_perm(temp, inst)
    mutate_assignment(inst, temp, pm1, rng, max_open)
    mutate_perm(temp, pm2, rng)
    _repair_capacity(inst, temp, max_open=max_open)
    evaluate(inst, temp, delta=delta)
    return temp


def _adopt(dst: Chromosome, src: Chromosome) -> None:
    dst.assignment_hub = dict(src.assignment_hub)
    dst.permutation = list(src.permutation)
    dst.fitness = src.fitness
    dst.objective = src.objective
    dst.penalty = src.penalty
    dst.open_hubs = None if src.open_hubs is None else list(src.open_hubs)
    dst.routes = None if src.routes is None else {h: [list(r) for r in rs] for h, rs in src.routes.items()}
    dst.feasible = src.feasible
    dst.violations = None if src.violations is None else list(src.violations)
    dst.costs = None if src.costs is None else dict(src.costs)


def _route_swap(inst: HLRPInstance, ch: Chromosome, delta: float) -> bool:
    best = ch.fitness
    n = len(ch.permutation)
    for i in range(n):
        for j in range(i + 1, n):
            cand = ch.copy()
            cand.permutation[i], cand.permutation[j] = cand.permutation[j], cand.permutation[i]
            evaluate(inst, cand, delta=delta)
            if cand.fitness < best - 1e-9:
                _adopt(ch, cand)
                return True
    return False


def _route_insert(inst: HLRPInstance, ch: Chromosome, delta: float) -> bool:
    best = ch.fitness
    n = len(ch.permutation)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            cand = ch.copy()
            node = cand.permutation.pop(i)
            cand.permutation.insert(j, node)
            evaluate(inst, cand, delta=delta)
            if cand.fitness < best - 1e-9:
                _adopt(ch, cand)
                return True
    return False


def _hub_close(inst: HLRPInstance, ch: Chromosome, delta: float) -> bool:
    open_hubs = sorted(set(ch.assignment_hub.values()))
    if len(open_hubs) <= 1:
        return False
    best = ch.fitness
    for close_h in open_hubs:
        remain = [h for h in open_hubs if h != close_h]
        cand = ch.copy()
        for c, h in list(cand.assignment_hub.items()):
            if h == close_h:
                cand.assignment_hub[c] = _nearest_hub(inst, c, remain)
        cand.assignment_hub = _normalize_assignment(inst, cand.assignment_hub)
        _rebuild_perm(cand, inst)
        evaluate(inst, cand, delta=delta)
        if cand.fitness < best - 1e-9:
            _adopt(ch, cand)
            return True
    return False


def _hub_open(inst: HLRPInstance, ch: Chromosome, delta: float, max_open: int) -> bool:
    open_hubs = sorted(set(ch.assignment_hub.values()))
    if len(open_hubs) >= max_open:
        return False
    closed = [i for i in range(inst.n) if i not in open_hubs]
    if not closed:
        return False
    best = ch.fitness
    for new_h in closed:
        cand = ch.copy()
        changed = False
        for c in list(cand.assignment_hub.keys()):
            new_cost = _assign_cost(inst, c, new_h)
            old_cost = _assign_cost(inst, c, cand.assignment_hub[c])
            if new_cost <= old_cost * 0.95:
                cand.assignment_hub[c] = new_h
                changed = True
        if not changed:
            continue
        cand.assignment_hub = _normalize_assignment(inst, cand.assignment_hub)
        _rebuild_perm(cand, inst)
        evaluate(inst, cand, delta=delta)
        if cand.fitness < best - 1e-9:
            _adopt(ch, cand)
            return True
    return False


def _hub_swap(inst: HLRPInstance, ch: Chromosome, delta: float) -> bool:
    open_hubs = sorted(set(ch.assignment_hub.values()))
    closed = [i for i in range(inst.n) if i not in open_hubs]
    if not open_hubs or not closed:
        return False
    best = ch.fitness
    for old_h in open_hubs:
        remain = [h for h in open_hubs if h != old_h]
        for new_h in closed:
            cand = ch.copy()
            for c, h in list(cand.assignment_hub.items()):
                if h == old_h:
                    cand.assignment_hub[c] = new_h
                else:
                    old_cost = _assign_cost(inst, c, h)
                    new_cost = _assign_cost(inst, c, new_h)
                    if new_cost <= old_cost * 0.9:
                        cand.assignment_hub[c] = new_h
            cand.assignment_hub = _normalize_assignment(inst, cand.assignment_hub)
            _rebuild_perm(cand, inst)
            evaluate(inst, cand, delta=delta)
            if cand.fitness < best - 1e-9:
                _adopt(ch, cand)
                return True
    return False


def local_search(inst: HLRPInstance, ch: Chromosome, delta: float, max_open: int, max_passes: int) -> Chromosome:
    evaluate(inst, ch, delta=delta)
    for _ in range(max_passes):
        changed = False
        changed |= _route_swap(inst, ch, delta)
        changed |= _route_insert(inst, ch, delta)
        changed |= _hub_close(inst, ch, delta)
        changed |= _hub_open(inst, ch, delta, max_open)
        changed |= _hub_swap(inst, ch, delta)
        if not changed:
            break
    _repair_capacity(inst, ch, max_open=max_open)
    evaluate(inst, ch, delta=delta)
    return ch



def solve_ga(inst: HLRPInstance, pop_size: int, max_gen: int, max_no_imp: int, pc: float, pm1: float, pm2: float, ls_passes: int, max_open: int, delta: float, seed: int, verbose: int, time_limit: float = 0.0) -> Dict[str, object]:
    rng = random.Random(seed)
    t0 = time.time()
    deadline = (t0 + float(time_limit)) if float(time_limit) > 0 else None
    pop = [random_chromosome(inst, rng, max_open=max_open) for _ in range(pop_size)]
    init_wall = time.time() - t0
    best = min(pop, key=lambda c: c.fitness).copy()
    feasible_pool = [c for c in pop if c.feasible]
    best_feasible = min(feasible_pool, key=lambda c: c.objective).copy() if feasible_pool else None

    first_feasible_obj = None
    best_feasible_obj = None
    time_to_first_feasible = None
    time_to_best_feasible = None
    first_feasible_step = None
    best_feasible_step = None
    improvement_history: List[Dict[str, object]] = []
    if best_feasible is not None:
        first_feasible_obj = float(best_feasible.objective)
        best_feasible_obj = float(best_feasible.objective)
        time_to_first_feasible = float(init_wall)
        time_to_best_feasible = float(init_wall)
        first_feasible_step = 0
        best_feasible_step = 0
        improvement_history.append({"t_s": float(init_wall), "step": 0, "obj": float(best_feasible.objective)})

    no_imp = 0
    history = []
    stopped_by_time = False
    for gen in range(1, max_gen + 1):
        if deadline is not None and time.time() >= deadline:
            stopped_by_time = True
            break
        for _ in range(pop_size):
            if deadline is not None and time.time() >= deadline:
                stopped_by_time = True
                break
            p1 = rank_selection(pop, rng)
            p2 = rank_selection(pop, rng)
            child = make_offspring(inst, p1, p2, rng, max_open, pc, pm1, pm2, delta)
            child = local_search(inst, child, delta, max_open, ls_passes)
            worst_idx = max(range(len(pop)), key=lambda i: pop[i].fitness)
            if child.fitness < pop[worst_idx].fitness - 1e-9:
                pop[worst_idx] = child.copy()

        cur = min(pop, key=lambda c: c.fitness)
        if cur.fitness < best.fitness - 1e-9:
            best = cur.copy()
            no_imp = 0
        else:
            no_imp += 1

        feasible_now = [c for c in pop if c.feasible]
        if feasible_now:
            cur_feas = min(feasible_now, key=lambda c: c.objective)
            if best_feasible is None or cur_feas.objective < best_feasible.objective - 1e-9:
                best_feasible = cur_feas.copy()
                now_t = float(time.time() - t0)
                if first_feasible_obj is None:
                    first_feasible_obj = float(cur_feas.objective)
                    time_to_first_feasible = now_t
                    first_feasible_step = gen
                best_feasible_obj = float(cur_feas.objective)
                time_to_best_feasible = now_t
                best_feasible_step = gen
                improvement_history.append({"t_s": now_t, "step": int(gen), "obj": float(cur_feas.objective)})

        hist_ref = best_feasible if best_feasible is not None else best
        history.append({'gen': gen, 'best_fit': float(hist_ref.fitness), 'obj': float(hist_ref.objective), 'pen': float(hist_ref.penalty), 'hubs': list(hist_ref.open_hubs or [])})
        if verbose >= 2 or (verbose >= 1 and gen % 10 == 0):
            log_ref = best_feasible if best_feasible is not None else best
            print(f"[GA] gen={gen:03d} best_fit={log_ref.fitness:.6f} obj={log_ref.objective:.6f} pen={log_ref.penalty:.6f} open={log_ref.open_hubs}")
        if deadline is None and no_imp >= max_no_imp:
            break

    wall = time.time() - t0
    final_ch = (best_feasible.copy() if best_feasible is not None else best.copy())
    _repair_capacity(inst, final_ch, max_open=max_open)
    evaluate(inst, final_ch, delta=delta)
    final_solution = _solution_from_chromosome(inst, final_ch)
    final_res = inst.evaluate_solution(final_solution)
    if final_res.feasible:
        if best_feasible is None or final_ch.objective < best_feasible.objective - 1e-9:
            best_feasible = final_ch.copy()
            now_t = float(wall)
            if first_feasible_obj is None:
                first_feasible_obj = float(final_ch.objective)
                time_to_first_feasible = now_t
                first_feasible_step = len(history)
            best_feasible_obj = float(final_ch.objective)
            time_to_best_feasible = now_t
            best_feasible_step = len(history)
            improvement_history.append({"t_s": now_t, "step": int(len(history)), "obj": float(final_ch.objective)})
        else:
            best_feasible = final_ch.copy()

    chosen = best_feasible.copy() if best_feasible is not None else best.copy()
    solution = _solution_from_chromosome(inst, chosen)
    res = inst.evaluate_solution(solution)
    best = chosen
    assign = {str(k): int(v) for k, v in solution['assignment'].items()}  # type: ignore[index]
    tours = {str(h): [[int(x) for x in rt] for rt in rs] for h, rs in solution['routes'].items()}  # type: ignore[index]
    time_to_last_improvement = time_to_best_feasible
    last_improvement_step = best_feasible_step

    dataset_type = str(inst.instance_name).split('_')[1] if '_' in str(inst.instance_name) else None
    stop_reason = 'time_limit' if stopped_by_time else ('no_improve' if deadline is None and no_imp >= max_no_imp else 'iteration_cap')
    runtime_utilization = None if float(time_limit) <= 0 else float(min(1.0, float(wall) / max(float(time_limit), 1e-12)))
    return {
        'instance': inst.instance_name,
        'dataset_type': dataset_type,
        'method': 'GA',
        'seed': int(seed),
        'time_limit_s': float(time_limit) if float(time_limit) > 0 else None,
        'total_wall_s': float(wall),
        'stop_reason': stop_reason,
        'final_feasible': bool(res.feasible),
        'final_obj': float(res.total_cost) if res.feasible else None,
        'source': '',
        'objective': float(res.total_cost if res.feasible else best.objective),
        'fitness': float(best.fitness),
        'hubs': list(best.open_hubs or []),
        'assign': assign,
        'tours': tours,
        'costs': {
            'route': float(getattr(res, 'local_route_cost', best.costs.get('route', 0.0) if best.costs else 0.0)),
            'assignment': float(getattr(res, 'assignment_cost', best.costs.get('assignment', 0.0) if best.costs else 0.0)),
            'interhub': float(getattr(res, 'inter_hub_flow_cost', best.costs.get('interhub', 0.0) if best.costs else 0.0)),
            'vehicle': float(getattr(res, 'vehicle_fixed_cost', best.costs.get('vehicle', 0.0) if best.costs else 0.0)),
            'hub': float(getattr(res, 'hub_fixed_cost', best.costs.get('hub', 0.0) if best.costs else 0.0)),
        },
        'perf': {
            'algorithm': 'GA',
            'seed': int(seed),
            'instance': inst.instance_name,
            'step_unit': 'generation',
            'stopped_by_time': bool(stopped_by_time),
            'stop_reason': stop_reason,
            'time_limit_s': float(time_limit) if float(time_limit) > 0 else None,
            'total_wall_s': float(wall),
            'runtime_utilization': runtime_utilization,
            'first_feasible_obj': None if first_feasible_obj is None else float(first_feasible_obj),
            'best_feasible_obj': None if best_feasible_obj is None else float(best_feasible_obj),
            'time_to_first_feasible_s': None if time_to_first_feasible is None else float(time_to_first_feasible),
            'time_to_best_feasible_s': None if time_to_best_feasible is None else float(time_to_best_feasible),
            'time_to_last_improvement_s': None if time_to_last_improvement is None else float(time_to_last_improvement),
            'first_feasible_step': first_feasible_step,
            'best_feasible_step': best_feasible_step,
            'last_improvement_step': last_improvement_step,
            'num_improvements': int(len(improvement_history)),
            'final_feasible': bool(res.feasible),
        },
        'improvement_history': improvement_history,
        'ga_settings': {
            'seed': seed,
            'pop_size': pop_size,
            'max_gen': max_gen,
            'time_limit': float(time_limit),
            'max_no_imp': max_no_imp,
            'pc': pc,
            'pm1': pm1,
            'pm2': pm2,
            'ls_passes': ls_passes,
            'max_open': max_open,
            'delta': delta,
        },
        'ga_stats': {
            'wall_s': float(wall),
            'generations': len(history),
            'stopped_by_time': bool(stopped_by_time),
            'history': history,
        },
        'feasible': bool(res.feasible),
        'violations': list(res.violations),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='GA baseline for the unified HLRP benchmark')
    ap.add_argument('--instance_json', type=str, required=True)
    ap.add_argument('--out_json', type=str, required=True)
    ap.add_argument('--pop_size', type=int, default=40)
    ap.add_argument('--max_gen', type=int, default=80)
    ap.add_argument('--max_no_imp', type=int, default=20)
    ap.add_argument('--pc', type=float, default=0.8)
    ap.add_argument('--pm1', type=float, default=0.7)
    ap.add_argument('--pm2', type=float, default=0.9)
    ap.add_argument('--ls_passes', type=int, default=1)
    ap.add_argument('--max_open', type=int, default=6)
    ap.add_argument('--delta', type=float, default=10000.0)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--verbose', type=int, default=2)
    ap.add_argument('--time_limit', type=float, default=0.0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    inst = load_instance(args.instance_json)
    out = solve_ga(
        inst,
        pop_size=args.pop_size,
        max_gen=args.max_gen,
        max_no_imp=args.max_no_imp,
        pc=args.pc,
        pm1=args.pm1,
        pm2=args.pm2,
        ls_passes=args.ls_passes,
        max_open=args.max_open,
        delta=args.delta,
        seed=args.seed,
        verbose=args.verbose,
        time_limit=args.time_limit,
    )
    out['source'] = args.instance_json
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[GA] done feasible={out['feasible']} obj={out['objective']:.6f} hubs={out['hubs']}")
    print(f"[GA] saved -> {args.out_json}")


if __name__ == '__main__':
    main()
