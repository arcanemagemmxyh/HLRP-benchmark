# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _import_common():
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent / 'evaluator' / 'hlrp_instance.py',
        Path.cwd() / 'src' / 'evaluator' / 'hlrp_instance.py',
        Path.cwd() / 'hlrp_instance.py',
        here.parent / 'hlrp_instance.py',
        Path.cwd() / 'hlrp_instance_v2.py',
        here.parent / 'hlrp_instance_v2.py',
    ]
    for f in candidates:
        if not f.exists():
            continue
        spec = importlib.util.spec_from_file_location('_hlrp_common_runtime_sa', f)
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod.HLRPInstance, mod.load_instance
    import hlrp_instance as mod  # type: ignore
    return mod.HLRPInstance, mod.load_instance


HLRPInstance, load_instance = _import_common()


@dataclass
class State:
    open_hubs: List[int]
    assignment: Dict[int, int]  # customers only, hubs omitted
    order: Dict[int, List[int]]

    def copy(self) -> 'State':
        return State(
            open_hubs=list(self.open_hubs),
            assignment=dict(self.assignment),
            order={h: list(v) for h, v in self.order.items()},
        )


def node_load(inst: HLRPInstance, i: int) -> float:
    outb = getattr(inst, 'outbound_flow', None)
    inb = getattr(inst, 'inbound_flow', None)
    if outb is not None and inb is not None:
        return float(outb[i]) + float(inb[i])
    return float(getattr(inst, 'node_throughput', [0.0] * inst.n)[i])


def assignment_cost_of(inst: HLRPInstance, i: int, h: int) -> float:
    if i == h:
        return 0.0
    if hasattr(inst, 'assignment_cost_of'):
        return float(inst.assignment_cost_of(i, h))
    w = float(getattr(inst, 'assignment_weight', 0.0))
    d = float(inst.distance[i][h])
    tp = list(getattr(inst, 'node_throughput_normalized', [1.0] * inst.n))
    return w * float(tp[i]) * d


def customers_of(inst: HLRPInstance, open_hubs: Iterable[int]) -> List[int]:
    hub_set = set(open_hubs)
    return [i for i in range(inst.n) if i not in hub_set]


def hub_loads(inst: HLRPInstance, assignment: Mapping[int, int], open_hubs: Iterable[int]) -> Dict[int, float]:
    if hasattr(inst, 'compute_hub_loads'):
        return {int(k): float(v) for k, v in inst.compute_hub_loads(assignment, open_hubs).items()}
    loads = {int(h): 0.0 for h in open_hubs}
    for i, h in assignment.items():
        if i in loads:
            continue
        loads[h] = loads.get(h, 0.0) + node_load(inst, i)
    return loads


def ensure_order_consistency(state: State) -> None:
    open_set = set(state.open_hubs)
    # remove invalid hub keys
    state.order = {h: list(state.order.get(h, [])) for h in state.open_hubs}
    seen = set()
    for h in state.open_hubs:
        cleaned = []
        for c in state.order.get(h, []):
            if c in open_set:
                continue
            if state.assignment.get(c) != h:
                continue
            if c in seen:
                continue
            cleaned.append(c)
            seen.add(c)
        state.order[h] = cleaned
    for c, h in list(state.assignment.items()):
        if c in open_set:
            state.assignment.pop(c, None)
            continue
        if h not in open_set:
            state.assignment.pop(c, None)
            continue
        if c not in seen:
            state.order.setdefault(h, []).append(c)
            seen.add(c)


def build_routes(inst: HLRPInstance, state: State) -> Dict[int, List[List[int]]]:
    ensure_order_consistency(state)
    routes: Dict[int, List[List[int]]] = {h: [] for h in state.open_hubs}
    for h in state.open_hubs:
        seq = list(state.order.get(h, []))
        for k in range(0, len(seq), inst.q):
            chunk = seq[k:k + inst.q]
            if chunk:
                routes[h].append(chunk)
    return routes


def solution_from_state(inst: HLRPInstance, state: State) -> Dict[str, object]:
    routes = build_routes(inst, state)
    return {
        'open_hubs': list(state.open_hubs),
        'assignment': {str(c): int(h) for c, h in sorted(state.assignment.items())},
        'routes': {str(h): [list(r) for r in rs] for h, rs in routes.items()},
    }


def route_seq_cost(inst: HLRPInstance, hub: int, seq: Sequence[int]) -> float:
    if not seq:
        return 0.0
    c = float(inst.distance[hub][seq[0]])
    for a, b in zip(seq[:-1], seq[1:]):
        c += float(inst.distance[a][b])
    c += float(inst.distance[seq[-1]][hub])
    return c


def nearest_order(inst: HLRPInstance, hub: int, nodes: Sequence[int], rng: random.Random) -> List[int]:
    rem = list(nodes)
    if not rem:
        return []
    start = min(rem, key=lambda x: (inst.distance[hub][x], rng.random()))
    order = [start]
    rem.remove(start)
    cur = start
    while rem:
        nxt = min(rem, key=lambda x: (inst.distance[cur][x], rng.random()))
        rem.remove(nxt)
        order.append(nxt)
        cur = nxt
    return order


def improve_order_2opt(inst: HLRPInstance, hub: int, seq: List[int], max_passes: int = 2) -> List[int]:
    if len(seq) <= 3:
        return seq
    best = list(seq)
    best_cost = route_seq_cost(inst, hub, best)
    for _ in range(max_passes):
        improved = False
        n = len(best)
        for i in range(0, n - 2):
            for j in range(i + 2, n):
                cand = best[:i] + list(reversed(best[i:j])) + best[j:]
                c = route_seq_cost(inst, hub, cand)
                if c + 1e-9 < best_cost:
                    best = cand
                    best_cost = c
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best


def canonicalize(inst: HLRPInstance, state: State) -> None:
    state.open_hubs = sorted(set(int(h) for h in state.open_hubs))
    open_set = set(state.open_hubs)
    # Remove hubs from assignment
    for h in list(open_set):
        state.assignment.pop(h, None)
    # Reorder per hub by current order, then fill missing with nearest order
    ensure_order_consistency(state)
    for h in state.open_hubs:
        assigned = [c for c, hh in state.assignment.items() if hh == h]
        existing = state.order.get(h, [])
        tail = [c for c in assigned if c not in existing]
        if tail:
            tail = nearest_order(inst, h, tail, random.Random(0))
        seq = list(existing) + tail
        # improve chunkwise
        improved = []
        for k in range(0, len(seq), inst.q):
            improved.extend(improve_order_2opt(inst, h, seq[k:k + inst.q]))
        state.order[h] = improved


def greedy_open_hubs(inst: HLRPInstance, max_open: int) -> List[int]:
    # surrogate: hub fixed + assignment to nearest open hub
    tp = [node_load(inst, i) for i in range(inst.n)]
    mean_tp = sum(tp) / max(1, inst.n)
    scale = [x / mean_tp if mean_tp > 1e-12 else 1.0 for x in tp]

    def surrogate(hubs: List[int]) -> float:
        open_set = set(hubs)
        val = sum(inst.hub_fixed_cost[h] for h in hubs)
        for i in range(inst.n):
            if i in open_set:
                continue
            best = min(assignment_cost_of(inst, i, h) for h in hubs)
            # add crude route proxy
            best += 0.25 * min(float(inst.distance[i][h]) for h in hubs)
            val += best
        return float(val)

    single = [(surrogate([h]), h) for h in range(inst.n)]
    single.sort()
    hubs = [single[0][1]]
    best_val = single[0][0]
    improved = True
    while improved and len(hubs) < max_open:
        improved = False
        cand_best = None
        for h in range(inst.n):
            if h in hubs:
                continue
            val = surrogate(hubs + [h])
            if cand_best is None or val < cand_best[0]:
                cand_best = (val, h)
        if cand_best is not None and cand_best[0] + 1e-9 < best_val:
            best_val = cand_best[0]
            hubs.append(cand_best[1])
            improved = True
    return sorted(hubs)


def assign_customers_greedy(inst: HLRPInstance, open_hubs: List[int], max_open: int, rng: random.Random) -> State:
    open_hubs = sorted(set(open_hubs))
    assignment: Dict[int, int] = {}
    loads = {h: 0.0 for h in open_hubs}
    customers = [i for i in range(inst.n) if i not in set(open_hubs)]
    customers.sort(key=lambda i: node_load(inst, i), reverse=True)
    for c in customers:
        load_c = node_load(inst, c)
        feasible = [h for h in open_hubs if loads[h] + load_c <= inst.hub_capacity[h] + 1e-9]
        if not feasible and len(open_hubs) < max_open:
            open_hubs.append(c)
            open_hubs = sorted(set(open_hubs))
            loads.setdefault(c, 0.0)
            continue
        cand_hubs = feasible if feasible else list(open_hubs)
        best_h = min(cand_hubs, key=lambda h: (assignment_cost_of(inst, c, h), inst.distance[c][h], rng.random()))
        assignment[c] = best_h
        loads[best_h] = loads.get(best_h, 0.0) + load_c
    state = State(open_hubs=sorted(open_hubs), assignment=assignment, order={})
    for h in state.open_hubs:
        nodes = [c for c, hh in assignment.items() if hh == h]
        state.order[h] = nearest_order(inst, h, nodes, rng)
    repair_capacity(inst, state, max_open, rng)
    canonicalize(inst, state)
    return state


def repair_capacity(inst: HLRPInstance, state: State, max_open: int, rng: random.Random, max_rounds: int = 200) -> bool:
    canonicalize(inst, state)
    for _ in range(max_rounds):
        loads = hub_loads(inst, state.assignment, state.open_hubs)
        overloaded = [h for h in state.open_hubs if loads.get(h, 0.0) - inst.hub_capacity[h] > 1e-9]
        if not overloaded:
            return True
        changed = False
        for h in sorted(overloaded, key=lambda x: loads.get(x, 0.0) - inst.hub_capacity[x], reverse=True):
            custs = [c for c, hh in state.assignment.items() if hh == h]
            if not custs:
                continue
            slack_hubs = [g for g in state.open_hubs if g != h and inst.hub_capacity[g] - loads.get(g, 0.0) > 1e-9]
            best_move = None
            best_key = None
            for c in custs:
                load_c = node_load(inst, c)
                for g in slack_hubs:
                    if loads.get(g, 0.0) + load_c > inst.hub_capacity[g] + 1e-9:
                        continue
                    delta = assignment_cost_of(inst, c, g) - assignment_cost_of(inst, c, h)
                    key = (delta, -load_c, inst.distance[c][g])
                    if best_key is None or key < best_key:
                        best_key = key
                        best_move = (c, g)
            if best_move is not None:
                c, g = best_move
                state.assignment[c] = g
                if c in state.order.get(h, []):
                    state.order[h].remove(c)
                state.order.setdefault(g, []).append(c)
                changed = True
                break
            if len(state.open_hubs) < max_open:
                # promote heavy customer to new hub
                c = max(custs, key=lambda x: (node_load(inst, x), assignment_cost_of(inst, x, h)))
                state.assignment.pop(c, None)
                if c in state.order.get(h, []):
                    state.order[h].remove(c)
                state.open_hubs.append(c)
                state.open_hubs = sorted(set(state.open_hubs))
                state.order.setdefault(c, [])
                # migrate nearest customers to new hub while there is capacity
                loads = hub_loads(inst, state.assignment, state.open_hubs)
                cap_new = inst.hub_capacity[c] - loads.get(c, 0.0)
                cand = [x for x in list(state.assignment) if state.assignment[x] == h]
                cand.sort(key=lambda x: (assignment_cost_of(inst, x, c) - assignment_cost_of(inst, x, h), inst.distance[x][c]))
                for x in cand:
                    load_x = node_load(inst, x)
                    if load_x <= cap_new + 1e-9:
                        state.assignment[x] = c
                        if x in state.order.get(h, []):
                            state.order[h].remove(x)
                        state.order.setdefault(c, []).append(x)
                        cap_new -= load_x
                changed = True
                break
        if not changed:
            break
        canonicalize(inst, state)
    loads = hub_loads(inst, state.assignment, state.open_hubs)
    return all(loads.get(h, 0.0) <= inst.hub_capacity[h] + 1e-9 for h in state.open_hubs)


def evaluate_state(inst: HLRPInstance, state: State) -> Tuple[float, object, Dict[str, object]]:
    sol = solution_from_state(inst, state)
    ev = inst.evaluate_solution(sol)
    return float(ev.total_cost), ev, sol


def initial_state(inst: HLRPInstance, max_open: int, rng: random.Random) -> State:
    hubs = greedy_open_hubs(inst, max_open)
    return assign_customers_greedy(inst, hubs, max_open, rng)


def infer_default_max_open(inst: HLRPInstance, instance_json: str = '') -> int:
    name = str(getattr(inst, 'instance_name', '')).lower()
    src = str(instance_json).lower()
    if '_tight' in name or '/tight/' in src or '\\tight\\' in src or name.endswith('tight') or '_tt' in name or '/tt/' in src or '\\tt\\' in src or src.endswith('tt'):
        return min(inst.n, 6)
    return max(2, min(inst.n, max(4, int(round(math.sqrt(inst.n))))))


def estimate_initial_temperature(
    inst: HLRPInstance,
    state: State,
    curr_val: float,
    curr_ev: object,
    max_open: int,
    rng: random.Random,
    samples: int = 24,
    target_accept: float = 0.8,
) -> float:
    deltas: List[float] = []
    ops = [neighbor_reassign, neighbor_open_close, neighbor_swap_hub, neighbor_route]
    for _ in range(samples):
        op = rng.choice(ops)
        cand = op(inst, state, max_open, rng)
        cand_val, cand_ev, _ = evaluate_state(inst, cand)
        if cand_ev.feasible and curr_ev.feasible:
            delta = cand_val - curr_val
            if delta > 1e-9 and math.isfinite(delta):
                deltas.append(delta)
    if deltas:
        avg_uphill = sum(deltas) / len(deltas)
        target_accept = min(max(target_accept, 1e-3), 1.0 - 1e-3)
        return max(1e-6, -avg_uphill / math.log(target_accept))
    base = curr_val if math.isfinite(curr_val) and curr_val > 1e-9 else 1e6
    return max(100.0, 0.02 * base)


def neighbor_reassign(inst: HLRPInstance, state: State, max_open: int, rng: random.Random) -> State:
    s = state.copy()
    customers = list(s.assignment.keys())
    if not customers:
        return s
    c = rng.choice(customers)
    old_h = s.assignment[c]
    loads = hub_loads(inst, s.assignment, s.open_hubs)
    load_c = node_load(inst, c)
    cand = [h for h in s.open_hubs if h != old_h and loads.get(h, 0.0) + load_c <= inst.hub_capacity[h] + 1e-9]
    if cand:
        new_h = min(cand, key=lambda h: (assignment_cost_of(inst, c, h), rng.random()))
        s.assignment[c] = new_h
        if c in s.order.get(old_h, []):
            s.order[old_h].remove(c)
        pos = rng.randrange(len(s.order.get(new_h, [])) + 1)
        s.order.setdefault(new_h, []).insert(pos, c)
    elif len(s.open_hubs) < max_open:
        # open new hub at c
        s.assignment.pop(c, None)
        if c in s.order.get(old_h, []):
            s.order[old_h].remove(c)
        s.open_hubs.append(c)
        s.open_hubs = sorted(set(s.open_hubs))
        s.order.setdefault(c, [])
        # move a few nearest nodes to the new hub
        old_nodes = [x for x, hh in s.assignment.items() if hh == old_h]
        cap = inst.hub_capacity[c]
        for x in sorted(old_nodes, key=lambda x: (assignment_cost_of(inst, x, c), inst.distance[x][c]))[:max(1, min(4, len(old_nodes)) )]:
            lx = node_load(inst, x)
            if lx <= cap + 1e-9:
                s.assignment[x] = c
                if x in s.order.get(old_h, []):
                    s.order[old_h].remove(x)
                s.order.setdefault(c, []).append(x)
                cap -= lx
    repair_capacity(inst, s, max_open, rng)
    canonicalize(inst, s)
    return s


def neighbor_open_close(inst: HLRPInstance, state: State, max_open: int, rng: random.Random) -> State:
    s = state.copy()
    if rng.random() < 0.5 and len(s.open_hubs) < max_open:
        custs = list(s.assignment.keys())
        if custs:
            c = rng.choice(custs)
            old_h = s.assignment[c]
            s.assignment.pop(c, None)
            if c in s.order.get(old_h, []):
                s.order[old_h].remove(c)
            s.open_hubs.append(c)
            s.open_hubs = sorted(set(s.open_hubs))
            s.order.setdefault(c, [])
    elif len(s.open_hubs) > 1:
        h = rng.choice(s.open_hubs)
        others = [g for g in s.open_hubs if g != h]
        for c in [x for x, hh in list(s.assignment.items()) if hh == h]:
            load_c = node_load(inst, c)
            feasible = [g for g in others if hub_loads(inst, s.assignment, s.open_hubs).get(g, 0.0) + load_c <= inst.hub_capacity[g] + 1e-9]
            target = min(feasible if feasible else others, key=lambda g: (assignment_cost_of(inst, c, g), inst.distance[c][g], rng.random()))
            s.assignment[c] = target
            if c in s.order.get(h, []):
                s.order[h].remove(c)
            s.order.setdefault(target, []).append(c)
        s.open_hubs.remove(h)
        s.order.pop(h, None)
    repair_capacity(inst, s, max_open, rng)
    canonicalize(inst, s)
    return s


def neighbor_swap_hub(inst: HLRPInstance, state: State, max_open: int, rng: random.Random) -> State:
    if not state.open_hubs or not state.assignment:
        return state.copy()
    s = state.copy()
    old_h = rng.choice(s.open_hubs)
    cand = rng.choice(list(s.assignment.keys()))
    old_nodes = [x for x, hh in s.assignment.items() if hh == old_h and x != cand]
    # close old_h, open cand
    s.open_hubs.remove(old_h)
    s.order.pop(old_h, None)
    s.open_hubs.append(cand)
    s.open_hubs = sorted(set(s.open_hubs))
    s.assignment.pop(cand, None)
    # reassign all customers from old_h to cand or others
    for x in old_nodes:
        s.assignment[x] = cand
    s.order[cand] = nearest_order(inst, cand, old_nodes, rng)
    repair_capacity(inst, s, max_open, rng)
    canonicalize(inst, s)
    return s


def neighbor_route(inst: HLRPInstance, state: State, max_open: int, rng: random.Random) -> State:
    s = state.copy()
    hubs = [h for h in s.open_hubs if len(s.order.get(h, [])) >= 2]
    if not hubs:
        return s
    h = rng.choice(hubs)
    seq = list(s.order[h])
    if len(seq) == 2:
        seq.reverse()
    else:
        i = rng.randrange(0, len(seq) - 1)
        j = rng.randrange(i + 1, len(seq))
        if rng.random() < 0.5:
            seq[i:j + 1] = reversed(seq[i:j + 1])
        else:
            x = seq.pop(j)
            seq.insert(i, x)
    s.order[h] = seq
    canonicalize(inst, s)
    return s


def local_search(inst: HLRPInstance, state: State, max_open: int, rng: random.Random, rounds: int = 40) -> State:
    best = state.copy()
    best_val, _, _ = evaluate_state(inst, best)
    ops = [neighbor_reassign, neighbor_open_close, neighbor_swap_hub, neighbor_route]
    no_imp = 0
    while no_imp < rounds:
        improved = False
        for _ in range(10):
            op = rng.choice(ops)
            cand = op(inst, best, max_open, rng)
            val, ev, _ = evaluate_state(inst, cand)
            if ev.feasible and val + 1e-9 < best_val:
                best = cand
                best_val = val
                improved = True
                no_imp = 0
                break
        if not improved:
            no_imp += 1
    return best



def sa_run(
    inst: HLRPInstance,
    seed: int,
    time_limit: float,
    max_open: int,
    inner_iters: int,
    temp0: float,
    cooling: float,
    stall_limit: int,
    reheat_gap: int,
    reheat_factor: float,
) -> Dict[str, object]:
    rng = random.Random(seed)
    t0 = time.time()
    curr = initial_state(inst, max_open, rng)
    curr = local_search(inst, curr, max_open, rng, rounds=24)
    curr_val, curr_ev, _ = evaluate_state(inst, curr)
    best = curr.copy()
    best_val = curr_val
    best_ev = curr_ev
    best_feas: Optional[State] = curr.copy() if curr_ev.feasible else None
    best_feas_val = curr_val if curr_ev.feasible else float('inf')
    best_feas_ev = curr_ev if curr_ev.feasible else None
    temp0_used = max(1e-6, temp0) if temp0 > 0 else estimate_initial_temperature(inst, curr, curr_val, curr_ev, max_open, rng)
    T = temp0_used
    iters = 0
    accepted = 0
    improved = 0
    stall = 0
    no_best = 0
    reheats = 0
    ops = [neighbor_reassign, neighbor_open_close, neighbor_swap_hub, neighbor_route]

    first_feasible_obj = None
    best_feasible_obj = None
    time_to_first_feasible = None
    time_to_best_feasible = None
    first_feasible_step = None
    best_feasible_step = None
    improvement_history: List[Dict[str, object]] = []
    if curr_ev.feasible:
        first_feasible_obj = float(curr_val)
        best_feasible_obj = float(curr_val)
        time_to_first_feasible = 0.0
        time_to_best_feasible = 0.0
        first_feasible_step = 0
        best_feasible_step = 0
        improvement_history.append({"t_s": 0.0, "step": 0, "obj": float(curr_val)})

    while True:
        if time_limit > 0 and time.time() - t0 >= time_limit:
            break
        level_improved = False
        for _ in range(inner_iters):
            if time_limit > 0 and time.time() - t0 >= time_limit:
                break
            iters += 1
            op = rng.choice(ops)
            cand = op(inst, curr, max_open, rng)
            if rng.random() < 0.35:
                cand = local_search(inst, cand, max_open, rng, rounds=2)
            cand_val, cand_ev, _ = evaluate_state(inst, cand)
            delta = cand_val - curr_val
            accept = False
            if cand_ev.feasible and not curr_ev.feasible:
                accept = True
            elif cand_ev.feasible and curr_ev.feasible:
                if delta <= 0:
                    accept = True
                elif T > 1e-12 and rng.random() < math.exp(-delta / T):
                    accept = True
            if cand_ev.feasible and cand_val + 1e-9 < best_feas_val:
                best_feas = cand.copy()
                best_feas_val = cand_val
                best_feas_ev = cand_ev
                best = cand.copy()
                best_val = cand_val
                best_ev = cand_ev
                improved += 1
                level_improved = True
                stall = 0
                no_best = 0
                now_t = float(time.time() - t0)
                if first_feasible_obj is None:
                    first_feasible_obj = float(cand_val)
                    time_to_first_feasible = now_t
                    first_feasible_step = iters
                best_feasible_obj = float(cand_val)
                time_to_best_feasible = now_t
                best_feasible_step = iters
                improvement_history.append({"t_s": now_t, "step": int(iters), "obj": float(cand_val)})
            if accept:
                curr, curr_val, curr_ev = cand, cand_val, cand_ev
                accepted += 1
            else:
                stall += 1
                no_best += 1
        if not level_improved:
            stall += 1
            no_best += inner_iters
        T *= cooling
        trigger_reheat = False
        if T < 1e-6:
            trigger_reheat = True
        if stall >= stall_limit:
            trigger_reheat = True
        if reheat_gap > 0 and no_best >= reheat_gap:
            trigger_reheat = True
        if trigger_reheat and (time_limit <= 0 or time.time() - t0 < time_limit):
            base = best_feas.copy() if best_feas is not None else best.copy()
            curr = base.copy()
            curr_val, curr_ev, _ = evaluate_state(inst, curr)
            T = max(temp0_used * max(0.05, reheat_factor), 1e-6)
            stall = 0
            no_best = 0
            reheats += 1
            if best_feas is not None and rng.random() < 0.25:
                temp0_used = max(temp0_used, estimate_initial_temperature(inst, best_feas, best_feas_val, best_feas_ev, max_open, rng))
        if T < 1e-9:
            T = max(temp0_used * max(0.05, reheat_factor), 1e-6)
            reheats += 1

    final_best = best_feas.copy() if best_feas is not None else best.copy()
    final_best = local_search(inst, final_best, max_open, rng, rounds=12)
    final_val, final_ev, final_sol = evaluate_state(inst, final_best)
    if final_ev.feasible:
        best = final_best
        best_val = final_val
        best_ev = final_ev
        best_sol = final_sol
    elif best_feas is not None and best_feas_ev is not None:
        best = best_feas.copy()
        best_val, best_ev, best_sol = evaluate_state(inst, best)
    else:
        best = final_best
        best_val, best_ev, best_sol = final_val, final_ev, final_sol

    if best_ev.feasible and (best_feasible_obj is None or best_val < best_feasible_obj - 1e-9):
        now_t = float(time.time() - t0)
        if first_feasible_obj is None:
            first_feasible_obj = float(best_val)
            time_to_first_feasible = now_t
            first_feasible_step = iters
        best_feasible_obj = float(best_val)
        time_to_best_feasible = now_t
        best_feasible_step = iters
        improvement_history.append({"t_s": now_t, "step": int(iters), "obj": float(best_val)})

    return {
        'best_state': best,
        'best_solution': best_sol,
        'best_eval': best_ev,
        'stats': {
            'seed': seed,
            'iters': iters,
            'accepted': accepted,
            'improvements': improved,
            'reheats': reheats,
            'wall_s': time.time() - t0,
            'temp0_arg': temp0,
            'temp0_used': temp0_used,
            'cooling': cooling,
            'inner_iters': inner_iters,
            'max_open': max_open,
            'reheat_gap': reheat_gap,
            'reheat_factor': reheat_factor,
            'best_feasible_found': bool(best_feas is not None),
        },
        'perf': {
            'algorithm': 'SA',
            'seed': int(seed),
            'instance': getattr(inst, 'instance_name', ''),
            'step_unit': 'sa_iteration',
            'stopped_by_time': bool(time_limit > 0 and (time.time() - t0) >= time_limit - 1e-9),
            'time_limit_s': float(time_limit) if float(time_limit) > 0 else None,
            'total_wall_s': float(time.time() - t0),
            'first_feasible_obj': None if first_feasible_obj is None else float(first_feasible_obj),
            'best_feasible_obj': None if best_feasible_obj is None else float(best_feasible_obj),
            'time_to_first_feasible_s': None if time_to_first_feasible is None else float(time_to_first_feasible),
            'time_to_best_feasible_s': None if time_to_best_feasible is None else float(time_to_best_feasible),
            'time_to_last_improvement_s': None if time_to_best_feasible is None else float(time_to_best_feasible),
            'first_feasible_step': first_feasible_step,
            'best_feasible_step': best_feasible_step,
            'last_improvement_step': best_feasible_step,
            'num_improvements': int(len(improvement_history)),
            'final_feasible': bool(best_ev.feasible),
        },
        'improvement_history': improvement_history,
    }



def multi_start_sa(
    inst: HLRPInstance,
    seed: int,
    time_limit: float,
    restarts: int,
    max_open: int,
    inner_iters: int,
    temp0: float,
    cooling: float,
    stall_limit: int,
    reheat_gap: int,
    reheat_factor: float,
) -> Dict[str, object]:
    all_runs = []
    best = None
    t0 = time.time()
    improvement_history: List[Dict[str, object]] = []
    first_feasible_obj = None
    best_feasible_obj = None
    time_to_first_feasible = None
    time_to_best_feasible = None
    first_feasible_step = None
    best_feasible_step = None
    step_offset = 0
    for r in range(restarts):
        remaining = max(0.0, time_limit - (time.time() - t0)) if time_limit > 0 else 0.0
        per_run = remaining / max(1, restarts - r) if time_limit > 0 else 0.0
        run_offset = float(time.time() - t0)
        res = sa_run(inst, seed + 1000 * r, per_run, max_open, inner_iters, temp0, cooling, stall_limit, reheat_gap, reheat_factor)
        all_runs.append(res['stats'])
        for ev in res.get('improvement_history', []):
            ge = {'t_s': float(ev['t_s']) + run_offset, 'step': int(ev['step']) + step_offset, 'obj': float(ev['obj'])}
            if first_feasible_obj is None:
                first_feasible_obj = ge['obj']
                time_to_first_feasible = ge['t_s']
                first_feasible_step = ge['step']
            if best_feasible_obj is None or ge['obj'] < best_feasible_obj - 1e-9:
                best_feasible_obj = ge['obj']
                time_to_best_feasible = ge['t_s']
                best_feasible_step = ge['step']
                improvement_history.append(ge)
        step_offset += int(res['stats'].get('iters', 0))
        if best is None:
            best = res
        else:
            cur_eval = res['best_eval']
            best_eval = best['best_eval']
            choose = False
            if cur_eval.feasible and not best_eval.feasible:
                choose = True
            elif cur_eval.feasible == best_eval.feasible and cur_eval.total_cost < best_eval.total_cost:
                choose = True
            if choose:
                best = res
        if time_limit > 0 and time.time() - t0 >= time_limit:
            break
    assert best is not None
    total_wall = float(time.time() - t0)
    best['stats_all'] = {
        'restarts': len(all_runs),
        'total_wall_s': total_wall,
        'runs': all_runs,
    }
    stop_reason = 'time_limit' if bool(time_limit > 0 and total_wall >= time_limit - 1e-9) else 'iteration_cap'
    runtime_utilization = None if float(time_limit) <= 0 else float(min(1.0, total_wall / max(float(time_limit), 1e-12)))
    best['perf_all'] = {
        'algorithm': 'SA',
        'seed': int(seed),
        'instance': getattr(inst, 'instance_name', ''),
        'step_unit': 'sa_iteration',
        'stopped_by_time': bool(time_limit > 0 and total_wall >= time_limit - 1e-9),
        'stop_reason': stop_reason,
        'time_limit_s': float(time_limit) if float(time_limit) > 0 else None,
        'total_wall_s': total_wall,
        'runtime_utilization': runtime_utilization,
        'first_feasible_obj': first_feasible_obj,
        'best_feasible_obj': best_feasible_obj,
        'time_to_first_feasible_s': time_to_first_feasible,
        'time_to_best_feasible_s': time_to_best_feasible,
        'time_to_last_improvement_s': time_to_best_feasible,
        'first_feasible_step': first_feasible_step,
        'best_feasible_step': best_feasible_step,
        'last_improvement_step': best_feasible_step,
        'num_improvements': int(len(improvement_history)),
        'final_feasible': bool(best['best_eval'].feasible),
    }
    best['improvement_history_all'] = improvement_history
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description='SA baseline for the unified HLRP benchmark')
    ap.add_argument('--instance_json', required=True)
    ap.add_argument('--out_json', default='')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--time_limit', type=float, default=60.0)
    ap.add_argument('--restarts', type=int, default=5)
    ap.add_argument('--inner_iters', type=int, default=80)
    ap.add_argument('--temp0', type=float, default=0.0)
    ap.add_argument('--cooling', type=float, default=0.92)
    ap.add_argument('--stall_limit', type=int, default=25)
    ap.add_argument('--reheat_gap', type=int, default=400)
    ap.add_argument('--reheat_factor', type=float, default=0.30)
    ap.add_argument('--max_open', type=int, default=0)
    ap.add_argument('--verbose', type=int, default=1)
    args = ap.parse_args()

    inst = load_instance(args.instance_json)
    if args.max_open > 0:
        max_open = args.max_open
    else:
        max_open = infer_default_max_open(inst, args.instance_json)
    res = multi_start_sa(
        inst=inst,
        seed=args.seed,
        time_limit=args.time_limit,
        restarts=args.restarts,
        max_open=max_open,
        inner_iters=args.inner_iters,
        temp0=args.temp0,
        cooling=args.cooling,
        stall_limit=args.stall_limit,
        reheat_gap=args.reheat_gap,
        reheat_factor=args.reheat_factor,
    )
    best_state = res['best_state']
    best_eval = res['best_eval']
    best_sol = res['best_solution']
    perf_all = res.get('perf_all', {})
    instance_name = getattr(inst, 'instance_name', Path(args.instance_json).stem)
    dataset_type = str(instance_name).split('_')[1] if '_' in str(instance_name) else None
    payload = {
        'instance': instance_name,
        'dataset_type': dataset_type,
        'method': 'SA',
        'seed': int(args.seed),
        'time_limit_s': float(args.time_limit),
        'total_wall_s': float(perf_all.get('total_wall_s', 0.0)),
        'stop_reason': perf_all.get('stop_reason', 'other'),
        'final_feasible': bool(best_eval.feasible),
        'final_obj': float(best_eval.total_cost) if best_eval.feasible else None,
        'source': args.instance_json,
        'objective': float(best_eval.total_cost),
        'hubs': list(best_state.open_hubs),
        'assign': {str(c): int(h) for c, h in sorted(best_state.assignment.items())},
        'tours': {str(h): [[h] + list(rt) + [h] for rt in best_sol['routes'].get(str(h), [])] for h in best_state.open_hubs},
        'costs': {
            'route': float(best_eval.local_route_cost),
            'assignment': float(getattr(best_eval, 'assignment_cost', 0.0)),
            'interhub': float(best_eval.inter_hub_flow_cost),
            'vehicle': float(best_eval.vehicle_fixed_cost),
            'hub': float(best_eval.hub_fixed_cost),
        },
        'benchmark_params': {
            'alpha': float(getattr(inst, 'alpha', 0.75)),
            'q': int(getattr(inst, 'q', 5)),
            'F_vehicle': float(getattr(inst, 'vehicle_fixed_cost_param', 0.0)),
            'assignment_weight': float(getattr(inst, 'assignment_weight', 0.0)),
            'Lambda': [float(x) for x in getattr(inst, 'hub_capacity', [])],
            'Fhub': [float(x) for x in getattr(inst, 'hub_fixed_cost', [])],
        },
        'perf': perf_all,
        'improvement_history': res.get('improvement_history_all', []),
        'sa_settings': {
            'seed': args.seed,
            'time_limit': float(args.time_limit),
            'restarts': int(args.restarts),
            'inner_iters': int(args.inner_iters),
            'temp0_arg': float(args.temp0),
            'temp0_used': float(res['stats']['temp0_used']),
            'cooling': float(args.cooling),
            'stall_limit': int(args.stall_limit),
            'max_open': int(max_open),
        },
        'sa_stats': res['stats_all'],
        'feasible': bool(best_eval.feasible),
        'violations': list(best_eval.violations),
    }
    if args.verbose:
        print(
            f"[SA] best={payload['objective']:.6f} hubs={payload['hubs']} "
            f"feasible={payload['feasible']} temp0_used={payload['sa_settings']['temp0_used']:.6f}"
        )
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
