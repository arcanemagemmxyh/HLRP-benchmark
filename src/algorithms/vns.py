# -*- coding: utf-8 -*-
"""
Unified VNS for the integration benchmark.

This is a variable-neighborhood-search style heuristic adapted from the user's
existing VNS line, but rewritten to solve the unified benchmark used by the
integration comparison:
- variable number of hubs
- single allocation
- hub capacity
- q-bounded local closed tours
- hub fixed cost + assignment cost + inter-hub cost + local route cost

Usage example
python src/algorithms/vns.py --instance_json data/main_core/loose/ap20_loose_q5.json --time_limit 60 --seed 11 --out scratch/ap20_loose_vns.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_COMMON_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "evaluator"))
if _COMMON_DIR not in sys.path:
    sys.path.insert(0, _COMMON_DIR)

from hlrp_instance import load_instance, HLRPInstance  # type: ignore


@dataclass
class VNSSolution:
    open_hubs: List[int]
    assignment: Dict[int, int]
    routes: Dict[int, List[List[int]]]
    total_cost: float = float("inf")
    hub_cost: float = 0.0
    route_cost: float = 0.0
    assignment_cost: float = 0.0
    interhub_cost: float = 0.0
    vehicle_cost: float = 0.0
    feasible: bool = False
    violations: Optional[List[str]] = None


class UnifiedVNS:
    def __init__(self, inst: HLRPInstance, seed: int = 1, max_open: Optional[int] = None):
        self.inst = inst
        self.seed = int(seed)
        self.rng = random.Random(seed)
        self.n = inst.n
        self.nodes = list(range(self.n))
        self.max_open = max(1, min(self.n, int(max_open) if max_open is not None else max(4, min(self.n, 8))))
        self.tp = [float(inst.outbound_flow[i] + inst.inbound_flow[i]) for i in self.nodes]
        mean_tp = sum(self.tp) / max(1, self.n)
        if mean_tp <= 1e-12:
            mean_tp = 1.0
        self.tp_norm = [x / mean_tp for x in self.tp]
        self.assign_weight = float(getattr(inst, "assignment_weight", 0.0))
        self.hub_rank = self._build_hub_rank()
        self._deadline_s: Optional[float] = None
        self._call_deadline_s: Optional[float] = None

    def _now_s(self) -> float:
        return time.perf_counter()

    def _time_exceeded(self) -> bool:
        deadlines = [x for x in (self._deadline_s, self._call_deadline_s) if x is not None]
        return bool(deadlines) and self._now_s() >= min(deadlines) - 1e-9


    def _build_hub_rank(self) -> List[int]:
        scores = []
        for h in self.nodes:
            weighted = 0.0
            for i in self.nodes:
                if i == h:
                    continue
                weighted += self.tp_norm[i] * self.inst.distance[i][h]
            score = self.inst.hub_fixed_cost[h] + weighted
            scores.append((score, h))
        scores.sort()
        return [h for _, h in scores]

    def clone(self, sol: VNSSolution) -> VNSSolution:
        return copy.deepcopy(sol)

    def assignment_cost_of(self, customer: int, hub: int) -> float:
        if customer == hub:
            return 0.0
        if hasattr(self.inst, "assignment_cost_of"):
            return float(self.inst.assignment_cost_of(customer, hub))
        return float(self.assign_weight * self.tp_norm[customer] * self.inst.distance[customer][hub])

    def route_cost(self, hub: int, route: Sequence[int]) -> float:
        return float(self.inst.route_cost(hub, route))

    def improve_route(self, hub: int, route: List[int]) -> List[int]:
        if len(route) <= 2:
            return list(route)
        best = list(route)
        improved = True
        while improved:
            if self._time_exceeded():
                break
            improved = False
            best_cost = self.route_cost(hub, best)
            # relocate
            for i in range(len(best)):
                if self._time_exceeded():
                    return best
                for j in range(len(best)):
                    if self._time_exceeded():
                        return best
                    if i == j:
                        continue
                    cand = list(best)
                    node = cand.pop(i)
                    cand.insert(j, node)
                    c = self.route_cost(hub, cand)
                    if c + 1e-9 < best_cost:
                        best = cand
                        best_cost = c
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
            # 2-opt style reversal
            for i in range(len(best) - 1):
                if self._time_exceeded():
                    return best
                for j in range(i + 1, len(best)):
                    if self._time_exceeded():
                        return best
                    cand = best[:i] + list(reversed(best[i:j + 1])) + best[j + 1:]
                    c = self.route_cost(hub, cand)
                    if c + 1e-9 < best_cost:
                        best = cand
                        best_cost = c
                        improved = True
                        break
                if improved:
                    break
        return best

    def build_routes_from_assignment(self, open_hubs: Sequence[int], assignment: Dict[int, int]) -> Dict[int, List[List[int]]]:
        routes: Dict[int, List[List[int]]] = {int(h): [] for h in open_hubs}
        nodes_meta = getattr(self.inst, "nodes", []) or []
        open_set = set(int(h) for h in open_hubs)
        for h in open_hubs:
            if self._time_exceeded():
                break
            custs = [i for i, ah in assignment.items() if i not in open_set and ah == h]
            if not custs:
                continue
            if h < len(nodes_meta) and all(k in nodes_meta[h] for k in ("x", "y")):
                hx = float(nodes_meta[h].get("x", 0.0))
                hy = float(nodes_meta[h].get("y", 0.0))
                custs.sort(key=lambda c: math.atan2(float(nodes_meta[c].get("y", 0.0)) - hy,
                                                    float(nodes_meta[c].get("x", 0.0)) - hx))
            else:
                custs.sort(key=lambda c: (self.inst.distance[h][c], c))
            chunks = [custs[i:i + self.inst.q] for i in range(0, len(custs), self.inst.q)]
            routes[h] = [self.improve_route(h, list(chunk)) for chunk in chunks if chunk]
        return routes

    def evaluate(self, open_hubs: Sequence[int], assignment: Dict[int, int]) -> VNSSolution:
        open_hubs = sorted(set(int(h) for h in open_hubs))
        if not open_hubs:
            open_hubs = [self.hub_rank[0]]
        assign = {int(h): int(h) for h in open_hubs}
        for i, h in assignment.items():
            assign[int(i)] = int(h)
        for i in self.nodes:
            if i not in open_hubs and i not in assign:
                # conservative fallback
                assign[i] = min(open_hubs, key=lambda h: self.assignment_cost_of(i, h))
        routes = self.build_routes_from_assignment(open_hubs, assign)
        sol_dict = {
            "open_hubs": open_hubs,
            "assignment": {str(i): int(h) for i, h in assign.items() if i not in open_hubs},
            "routes": {str(h): routes.get(h, []) for h in open_hubs},
        }
        res = self.inst.evaluate_solution(sol_dict)
        return VNSSolution(
            open_hubs=open_hubs,
            assignment=assign,
            routes=routes,
            total_cost=float(res.total_cost),
            hub_cost=float(res.hub_fixed_cost),
            route_cost=float(res.local_route_cost),
            assignment_cost=float(getattr(res, "assignment_cost", 0.0)),
            interhub_cost=float(res.inter_hub_flow_cost),
            vehicle_cost=float(res.vehicle_fixed_cost),
            feasible=bool(res.feasible),
            violations=list(res.violations),
        )

    def greedy_assign(self, open_hubs: Sequence[int], allow_open_extra: bool = True) -> Tuple[List[int], Dict[int, int]]:
        open_list = sorted(set(int(h) for h in open_hubs))
        if not open_list:
            open_list = [self.hub_rank[0]]
        assignment: Dict[int, int] = {h: h for h in open_list}
        loads = {h: 0.0 for h in open_list}
        customers = [i for i in self.nodes if i not in open_list]
        customers.sort(key=lambda i: (-self.tp[i], i))
        for c in customers:
            if self._time_exceeded():
                break
            load = self.tp[c]
            best_h = None
            best_score = float("inf")
            best_over = float("inf")
            for h in open_list:
                over = max(0.0, loads[h] + load - self.inst.hub_capacity[h])
                score = self.assignment_cost_of(c, h) + 0.15 * self.inst.distance[c][h] + 1e7 * over
                if score < best_score - 1e-9:
                    best_score = score
                    best_h = h
                    best_over = over
            if best_h is None:
                best_h = open_list[0]
                best_over = float("inf")
            if best_over > 1e-9 and allow_open_extra and len(open_list) < self.max_open:
                open_list.append(c)
                open_list = sorted(set(open_list))
                assignment[c] = c
                loads.setdefault(c, 0.0)
            else:
                assignment[c] = int(best_h)
                loads[int(best_h)] += load
        self.capacity_repair(open_list, assignment)
        return sorted(set(open_list)), assignment

    def capacity_repair(self, open_hubs: List[int], assignment: Dict[int, int]) -> None:
        open_set = set(open_hubs)
        for h in open_hubs:
            assignment[h] = h
        max_iter = 5 * max(1, self.n)
        it = 0
        while it < max_iter:
            if self._time_exceeded():
                return
            it += 1
            loads = self.compute_loads(open_hubs, assignment)
            overloaded = [h for h in open_hubs if loads[h] - self.inst.hub_capacity[h] > 1e-9]
            if not overloaded:
                return
            h = max(overloaded, key=lambda x: loads[x] - self.inst.hub_capacity[x])
            custs = [i for i in self.nodes if i not in open_set and assignment.get(i) == h]
            if not custs:
                break
            custs.sort(key=lambda i: (-self.tp[i], -self.assignment_cost_of(i, h), i))
            moved = False
            for c in custs:
                if self._time_exceeded():
                    return
                load_c = self.tp[c]
                best_t = None
                best_delta = float("inf")
                for t in open_hubs:
                    if t == h:
                        continue
                    if loads[t] + load_c - self.inst.hub_capacity[t] > 1e-9:
                        continue
                    delta = self.assignment_cost_of(c, t) - self.assignment_cost_of(c, h)
                    delta += 0.10 * (self.inst.distance[c][t] - self.inst.distance[c][h])
                    if delta < best_delta:
                        best_delta = delta
                        best_t = t
                if best_t is not None:
                    assignment[c] = int(best_t)
                    loads[h] -= load_c
                    loads[int(best_t)] += load_c
                    moved = True
                    break
                if len(open_hubs) < self.max_open:
                    open_hubs.append(c)
                    open_hubs[:] = sorted(set(open_hubs))
                    open_set = set(open_hubs)
                    assignment[c] = c
                    moved = True
                    break
            if not moved:
                # fallback: open one cheap closed node from this overloaded cluster if possible
                if len(open_hubs) < self.max_open:
                    cand = min(custs, key=lambda i: self.inst.hub_fixed_cost[i])
                    open_hubs.append(cand)
                    open_hubs[:] = sorted(set(open_hubs))
                    open_set = set(open_hubs)
                    assignment[cand] = cand
                    moved = True
                else:
                    break
        # one last greedy re-route of remaining overloaded customers if possible
        loads = self.compute_loads(open_hubs, assignment)
        for h in list(open_hubs):
            while loads[h] - self.inst.hub_capacity[h] > 1e-9:
                if self._time_exceeded():
                    return
                custs = [i for i in self.nodes if i not in open_set and assignment.get(i) == h]
                if not custs:
                    break
                c = max(custs, key=lambda i: self.tp[i])
                feasible_targets = [t for t in open_hubs if t != h and loads[t] + self.tp[c] - self.inst.hub_capacity[t] <= 1e-9]
                if not feasible_targets:
                    break
                t = min(feasible_targets, key=lambda x: (self.assignment_cost_of(c, x), x))
                assignment[c] = t
                loads[h] -= self.tp[c]
                loads[t] += self.tp[c]

    def compute_loads(self, open_hubs: Iterable[int], assignment: Dict[int, int]) -> Dict[int, float]:
        loads = {int(h): 0.0 for h in open_hubs}
        open_set = set(int(h) for h in open_hubs)
        for i, h in assignment.items():
            if i in open_set:
                continue
            loads[int(h)] = loads.get(int(h), 0.0) + self.tp[i]
        return loads

    def build_from_hubs(self, open_hubs: Sequence[int]) -> VNSSolution:
        hubs, assign = self.greedy_assign(open_hubs, allow_open_extra=True)
        sol = self.evaluate(hubs, assign)
        if not sol.feasible:
            hubs2, assign2 = self.greedy_assign(hubs, allow_open_extra=True)
            sol = self.evaluate(hubs2, assign2)
        return sol

    def make_diversified_hubs(self, k: int, offset: int = 0) -> List[int]:
        rank = self.hub_rank[:]
        first = rank[offset % len(rank)]
        chosen = [first]
        while len(chosen) < k:
            best = None
            best_score = -float("inf")
            for cand in self.nodes:
                if cand in chosen:
                    continue
                min_d = min(self.inst.distance[cand][h] for h in chosen)
                desir = sum(self.tp_norm[i] * self.inst.distance[i][cand] for i in self.nodes)
                score = min_d + 0.15 * desir - 0.0001 * self.inst.hub_fixed_cost[cand]
                if score > best_score:
                    best_score = score
                    best = cand
            chosen.append(int(best))
        return sorted(chosen)

    def initial_solution(self) -> VNSSolution:
        candidates: List[VNSSolution] = []
        max_k = min(self.max_open, 4)
        for k in range(1, max_k + 1):
            if self._time_exceeded():
                break
            for offset in range(min(3, self.n)):
                if self._time_exceeded():
                    break
                hubs = self.make_diversified_hubs(k, offset=offset)
                candidates.append(self.build_from_hubs(hubs))
        if not candidates:
            return self.build_from_hubs([self.hub_rank[0]])
        best = min(candidates, key=lambda s: (float("inf") if not s.feasible else s.total_cost, s.total_cost))
        return best

    def try_reassign_local(self, sol: VNSSolution) -> VNSSolution:
        best = self.clone(sol)
        improved = True
        while improved:
            if self._time_exceeded():
                break
            improved = False
            loads = self.compute_loads(best.open_hubs, best.assignment)
            customers = [i for i in self.nodes if i not in best.open_hubs]
            customers.sort(key=lambda i: (-self.tp[i], i))
            for c in customers:
                if self._time_exceeded():
                    return best
                cur = best.assignment[c]
                for h in best.open_hubs:
                    if self._time_exceeded():
                        return best
                    if h == cur:
                        continue
                    if loads[h] + self.tp[c] - self.inst.hub_capacity[h] > 1e-9:
                        continue
                    new_assign = dict(best.assignment)
                    new_assign[c] = h
                    cand = self.evaluate(best.open_hubs, new_assign)
                    if cand.feasible and cand.total_cost + 1e-9 < best.total_cost:
                        best = cand
                        loads = self.compute_loads(best.open_hubs, best.assignment)
                        improved = True
                        break
                if improved:
                    break
        return best

    def local_search(self, sol: VNSSolution, time_limit_s: float, start_time: float,
                     call_budget_s: Optional[float] = None) -> Tuple[VNSSolution, Dict[str, int]]:
        old_call_deadline = self._call_deadline_s
        if call_budget_s is not None and call_budget_s > 0:
            call_deadline = self._now_s() + float(call_budget_s)
            self._call_deadline_s = call_deadline if old_call_deadline is None else min(old_call_deadline, call_deadline)
        try:
            return self._local_search_impl(sol, time_limit_s, start_time)
        finally:
            self._call_deadline_s = old_call_deadline

    def _local_search_impl(self, sol: VNSSolution, time_limit_s: float, start_time: float) -> Tuple[VNSSolution, Dict[str, int]]:
        ls_stats = {
            "reassign_improvements": 0,
            "close_improvements": 0,
            "open_improvements": 0,
            "swap_improvements": 0,
            "passes": 0,
        }
        if self._time_exceeded():
            return sol, ls_stats
        best0 = self.try_reassign_local(sol)
        best = best0
        if best.total_cost + 1e-9 < sol.total_cost:
            ls_stats["reassign_improvements"] += 1
        improved = True
        while improved and not self._time_exceeded():
            ls_stats["passes"] += 1
            improved = False
            current_best = best
            closed = [i for i in self.nodes if i not in best.open_hubs]

            if len(best.open_hubs) > 1:
                for h in list(best.open_hubs):
                    if self._time_exceeded():
                        return best, ls_stats
                    cand_hubs = [x for x in best.open_hubs if x != h]
                    cand = self.build_from_hubs(cand_hubs)
                    if self._time_exceeded():
                        return best, ls_stats
                    cand = self.try_reassign_local(cand)
                    if cand.feasible and cand.total_cost + 1e-9 < current_best.total_cost:
                        current_best = cand
                        improved = True
                        ls_stats["close_improvements"] += 1
                        break
                if improved:
                    best = current_best
                    continue

            if len(best.open_hubs) < self.max_open:
                for cand_h in closed:
                    if self._time_exceeded():
                        return best, ls_stats
                    cand_hubs = sorted(best.open_hubs + [cand_h])
                    cand = self.build_from_hubs(cand_hubs)
                    if self._time_exceeded():
                        return best, ls_stats
                    cand = self.try_reassign_local(cand)
                    if cand.feasible and cand.total_cost + 1e-9 < current_best.total_cost:
                        current_best = cand
                        improved = True
                        ls_stats["open_improvements"] += 1
                        break
                if improved:
                    best = current_best
                    continue

            for out_h in list(best.open_hubs):
                if self._time_exceeded():
                    return best, ls_stats
                for in_h in closed:
                    if self._time_exceeded():
                        return best, ls_stats
                    cand_hubs = sorted([h for h in best.open_hubs if h != out_h] + [in_h])
                    cand = self.build_from_hubs(cand_hubs)
                    if self._time_exceeded():
                        return best, ls_stats
                    cand = self.try_reassign_local(cand)
                    if cand.feasible and cand.total_cost + 1e-9 < current_best.total_cost:
                        current_best = cand
                        improved = True
                        ls_stats["swap_improvements"] += 1
                        break
                if improved:
                    break
            if improved:
                best = current_best
        return best, ls_stats

    def rebuild_from_seed(self, open_hubs: Sequence[int], seed_assignment: Optional[Dict[int, int]] = None,
                          allow_open_extra: bool = True) -> VNSSolution:
        open_list = sorted(set(int(h) for h in open_hubs))
        if not open_list:
            open_list = [self.hub_rank[0]]
        assignment: Dict[int, int] = {h: h for h in open_list}
        loads = {h: 0.0 for h in open_list}
        customers = [i for i in self.nodes if i not in open_list]
        # Start from "difficult" customers so perturbations survive the rebuild stage.
        customers.sort(key=lambda i: (-self.tp[i], -min(self.inst.distance[i][h] for h in open_list), i))
        for c in customers:
            if self._time_exceeded():
                break
            load = self.tp[c]
            preferred = None if seed_assignment is None else seed_assignment.get(c)
            best_h = None
            best_score = float("inf")
            best_over = float("inf")
            for h in open_list:
                over = max(0.0, loads[h] + load - self.inst.hub_capacity[h])
                score = self.assignment_cost_of(c, h) + 0.15 * self.inst.distance[c][h] + 1e7 * over
                if preferred == h:
                    score *= 0.88
                if score < best_score - 1e-9:
                    best_score = score
                    best_h = h
                    best_over = over
            if best_h is None:
                best_h = open_list[0]
                best_over = float("inf")
            if best_over > 1e-9 and allow_open_extra and len(open_list) < self.max_open:
                open_list.append(c)
                open_list = sorted(set(open_list))
                assignment[c] = c
                loads.setdefault(c, 0.0)
            else:
                assignment[c] = int(best_h)
                loads[int(best_h)] += load
        self.capacity_repair(open_list, assignment)
        return self.evaluate(open_list, assignment)

    def shake(self, sol: VNSSolution, k: int) -> VNSSolution:
        hubs = list(sol.open_hubs)
        assignment = {int(i): int(h) for i, h in sol.assignment.items()}
        for h in hubs:
            assignment[h] = h
        ops = max(2, 2 * max(1, k))
        for _ in range(ops):
            if self._time_exceeded():
                break
            closed = [i for i in self.nodes if i not in hubs]
            customers = [i for i in self.nodes if i not in hubs]
            customer_costs = []
            for c in customers:
                if self._time_exceeded():
                    break
                ah = assignment.get(c, hubs[0] if hubs else 0)
                customer_costs.append((self.assignment_cost_of(c, ah) + 0.15 * self.inst.distance[c][ah], c))
            customer_costs.sort(reverse=True)
            move = self.rng.choice(["swap", "open", "close", "reassign", "cluster_open"])
            if move == "swap" and hubs and closed:
                hub_loads = self.compute_loads(hubs, assignment)
                out_h = max(hubs, key=lambda h: (self.inst.hub_fixed_cost[h] + 0.3 * hub_loads.get(h, 0.0), h))
                in_h = customer_costs[0][1] if customer_costs else self.rng.choice(closed)
                if in_h not in hubs:
                    hubs = sorted([h for h in hubs if h != out_h] + [in_h])
                    for c in list(assignment):
                        if assignment.get(c) == out_h and c not in hubs:
                            assignment[c] = in_h
                    assignment[in_h] = in_h
                    if out_h in assignment and out_h not in hubs:
                        assignment.pop(out_h, None)
            elif move == "open" and closed and len(hubs) < self.max_open:
                new_h = customer_costs[0][1] if customer_costs else self.rng.choice(closed)
                if new_h not in hubs:
                    hubs = sorted(hubs + [new_h])
                    assignment[new_h] = new_h
                    batch = [c for _, c in customer_costs[:max(2, k + 1)] if c != new_h]
                    batch += [c for c in customers if c != new_h and self.inst.distance[c][new_h] < self.inst.distance[c][assignment.get(c, hubs[0])]]
                    seen = set()
                    for c in batch:
                        if self._time_exceeded():
                            break
                        if c in seen or c in hubs:
                            continue
                        seen.add(c)
                        assignment[c] = new_h
            elif move == "close" and len(hubs) > 1:
                hub_loads = self.compute_loads(hubs, assignment)
                out_h = min(hubs, key=lambda h: (hub_loads.get(h, 0.0), -self.inst.hub_fixed_cost[h], h))
                hubs = [h for h in hubs if h != out_h]
                for c in list(assignment):
                    if self._time_exceeded():
                        break
                    if c == out_h:
                        assignment.pop(c, None)
                    elif assignment.get(c) == out_h and c not in hubs:
                        assignment.pop(c, None)
            elif move == "reassign" and len(hubs) >= 2:
                batch = [c for _, c in customer_costs[:max(2, 2 * k)]]
                self.rng.shuffle(batch)
                for c in batch:
                    if self._time_exceeded():
                        break
                    cur = assignment.get(c)
                    alts = [h for h in hubs if h != cur]
                    if not alts:
                        continue
                    alt = min(alts, key=lambda h: (self.assignment_cost_of(c, h) + 0.05 * self.inst.distance[c][h], h))
                    assignment[c] = alt
            elif move == "cluster_open" and closed and len(hubs) < self.max_open:
                seed = customer_costs[0][1] if customer_costs else self.rng.choice(closed)
                nearby = sorted([c for c in customers if c != seed], key=lambda c: self.inst.distance[c][seed])[:max(2, 2 * k)]
                candidate_pool = [seed] + nearby
                new_h = min(candidate_pool, key=lambda x: self.inst.hub_fixed_cost[x])
                if new_h not in hubs:
                    hubs = sorted(hubs + [new_h])
                    assignment[new_h] = new_h
                for c in candidate_pool:
                    if self._time_exceeded():
                        break
                    if c not in hubs:
                        assignment[c] = new_h
        return self.rebuild_from_seed(hubs, assignment, allow_open_extra=True)

    def solve(self, kmax: int = 5, time_limit: float = 60.0) -> Tuple[VNSSolution, Dict[str, object]]:
        start = self._now_s()
        self._deadline_s = start + max(1.0, float(time_limit))
        init0 = self.initial_solution()
        init_obj = init0.total_cost
        init_feasible = init0.feasible

        improvement_history: List[Dict[str, object]] = []
        first_feasible_obj = None
        best_feasible_obj = None
        time_to_first_feasible = None
        time_to_best_feasible = None
        first_feasible_step = None
        best_feasible_step = None

        if init0.feasible:
            first_feasible_obj = float(init0.total_cost)
            best_feasible_obj = float(init0.total_cost)
            time_to_first_feasible = 0.0
            time_to_best_feasible = 0.0
            first_feasible_step = 0
            best_feasible_step = 0
            improvement_history.append({"t_s": 0.0, "step": 0, "obj": float(init0.total_cost)})

        if self._time_exceeded():
            wall_s = self._now_s() - start
            stats = {
                "iterations": 0,
                "improvements": 0,
                "current_updates": 0,
                "rejected_moves": 0,
                "shake_better_than_current": 0,
                "local_search_calls": 0,
                "local_search_stats": {
                    "reassign_improvements": 0,
                    "close_improvements": 0,
                    "open_improvements": 0,
                    "swap_improvements": 0,
                    "passes": 0,
                },
                "initial_objective": init_obj,
                "initial_feasible": init_feasible,
                "initial_objective_after_ls": init_obj,
                "first_improvement_iter": None,
                "wall_s": wall_s,
                "stopped_by_time": True,
                "stop_reason": "time_limit",
                "max_open": self.max_open,
                "hub_rank": self.hub_rank,
                "first_feasible_obj": first_feasible_obj,
                "best_feasible_obj": best_feasible_obj,
                "time_to_first_feasible_s": time_to_first_feasible,
                "time_to_best_feasible_s": time_to_best_feasible,
                "time_to_last_improvement_s": time_to_best_feasible,
                "first_feasible_step": first_feasible_step,
                "best_feasible_step": best_feasible_step,
                "last_improvement_step": best_feasible_step,
                "num_improvements": int(len(improvement_history)),
                "improvement_history": improvement_history,
            }
            self._deadline_s = None
            return init0, stats

        init_ls_budget = max(5.0, min(60.0, 0.15 * float(time_limit)))
        iter_ls_budget = max(2.0, min(20.0, 0.05 * float(time_limit)))
        best, init_ls_stats = self.local_search(init0, time_limit, start, call_budget_s=init_ls_budget)
        init_after_ls_obj = best.total_cost
        curr = self.clone(best)
        if best.feasible:
            now_t = self._now_s() - start
            if first_feasible_obj is None:
                first_feasible_obj = float(best.total_cost)
                best_feasible_obj = float(best.total_cost)
                time_to_first_feasible = now_t
                time_to_best_feasible = now_t
                first_feasible_step = 0
                best_feasible_step = 0
                improvement_history.append({"t_s": float(now_t), "step": 0, "obj": float(best.total_cost)})
            elif best.total_cost + 1e-9 < float(best_feasible_obj):
                best_feasible_obj = float(best.total_cost)
                time_to_best_feasible = now_t
                best_feasible_step = 0
                improvement_history.append({"t_s": float(now_t), "step": 0, "obj": float(best.total_cost)})

        k = 1
        iters = 0
        best_updates = 0
        current_updates = 0
        rejected_moves = 0
        shake_better_than_current = 0
        local_search_calls = 1
        ls_totals = dict(init_ls_stats)
        first_improvement_iter = None

        while not self._time_exceeded():
            iters += 1
            shaken = self.shake(curr, k)
            if shaken.feasible and shaken.total_cost + 1e-9 < curr.total_cost:
                shake_better_than_current += 1
            if self._time_exceeded():
                break
            cand, ls_stats = self.local_search(shaken, time_limit, start, call_budget_s=iter_ls_budget)
            local_search_calls += 1
            for kk, vv in ls_stats.items():
                ls_totals[kk] = ls_totals.get(kk, 0) + int(vv)
            if cand.feasible and cand.total_cost + 1e-9 < best.total_cost:
                best = self.clone(cand)
                curr = self.clone(cand)
                k = 1
                best_updates += 1
                current_updates += 1
                if first_improvement_iter is None:
                    first_improvement_iter = iters
                now_t = self._now_s() - start
                if first_feasible_obj is None:
                    first_feasible_obj = float(cand.total_cost)
                    time_to_first_feasible = now_t
                    first_feasible_step = iters
                best_feasible_obj = float(cand.total_cost)
                time_to_best_feasible = now_t
                best_feasible_step = iters
                improvement_history.append({"t_s": float(now_t), "step": int(iters), "obj": float(cand.total_cost)})
            elif cand.feasible and cand.total_cost + 1e-9 < curr.total_cost:
                curr = self.clone(cand)
                current_updates += 1
                k += 1
                if k > max(1, kmax):
                    k = 1
            else:
                rejected_moves += 1
                k += 1
                if k > max(1, kmax):
                    k = 1

        wall_s = self._now_s() - start
        stopped_by_time = bool(self._time_exceeded())
        stop_reason = "time_limit" if stopped_by_time else "other"
        stats = {
            "iterations": iters,
            "improvements": best_updates,
            "current_updates": current_updates,
            "rejected_moves": rejected_moves,
            "shake_better_than_current": shake_better_than_current,
            "local_search_calls": local_search_calls,
            "local_search_stats": ls_totals,
            "initial_objective": init_obj,
            "initial_feasible": init_feasible,
            "initial_objective_after_ls": init_after_ls_obj,
            "first_improvement_iter": first_improvement_iter,
            "wall_s": wall_s,
            "stopped_by_time": stopped_by_time,
            "stop_reason": stop_reason,
            "max_open": self.max_open,
            "hub_rank": self.hub_rank,
            "first_feasible_obj": first_feasible_obj,
            "best_feasible_obj": best_feasible_obj,
            "time_to_first_feasible_s": time_to_first_feasible,
            "time_to_best_feasible_s": time_to_best_feasible,
            "time_to_last_improvement_s": time_to_best_feasible,
            "first_feasible_step": first_feasible_step,
            "best_feasible_step": best_feasible_step,
            "last_improvement_step": best_feasible_step,
            "num_improvements": int(len(improvement_history)),
            "improvement_history": improvement_history,
        }
        self._deadline_s = None
        return best, stats



def solution_to_dict(sol: VNSSolution) -> Dict[str, object]:
    return {
        "open_hubs": list(sol.open_hubs),
        "assignment": {str(i): int(h) for i, h in sol.assignment.items() if i not in sol.open_hubs},
        "routes": {str(h): [list(r) for r in sol.routes.get(h, [])] for h in sol.open_hubs},
    }


def infer_dataset_type(instance_name: str) -> Optional[str]:
    parts = str(instance_name).lower().split("_")
    for token in parts:
        if token in {"loose", "tight"}:
            return token
        if token == "ll":
            return "loose"
        if token == "tt":
            return "tight"
    return None


def build_output(instance_json: str, inst: HLRPInstance, sol: VNSSolution, settings: Dict[str, object], stats: Dict[str, object]) -> Dict[str, object]:
    total_wall_s = float(stats.get("wall_s", 0.0))
    time_limit_s = float(settings.get("time_limit", 0.0))
    dataset_type = infer_dataset_type(inst.instance_name)
    stop_reason = stats.get("stop_reason", "time_limit" if stats.get("stopped_by_time", False) else "other")
    perf = {
        "algorithm": "VNS",
        "seed": int(settings.get("seed", 0)),
        "instance": inst.instance_name,
        "step_unit": "iteration",
        "stopped_by_time": bool(stats.get("stopped_by_time", False)),
        "time_limit_s": time_limit_s,
        "total_wall_s": total_wall_s,
        "runtime_utilization": (total_wall_s / time_limit_s) if time_limit_s > 1e-12 else None,
        "first_feasible_obj": stats.get("first_feasible_obj"),
        "best_feasible_obj": stats.get("best_feasible_obj"),
        "time_to_first_feasible_s": stats.get("time_to_first_feasible_s"),
        "time_to_best_feasible_s": stats.get("time_to_best_feasible_s"),
        "time_to_last_improvement_s": stats.get("time_to_last_improvement_s"),
        "first_feasible_step": stats.get("first_feasible_step"),
        "best_feasible_step": stats.get("best_feasible_step"),
        "last_improvement_step": stats.get("last_improvement_step"),
        "num_improvements": int(stats.get("num_improvements", len(stats.get("improvement_history", [])))),
    }
    return {
        "instance": inst.instance_name,
        "dataset_type": dataset_type,
        "method": "VNS",
        "seed": int(settings.get("seed", 0)),
        "time_limit_s": time_limit_s,
        "total_wall_s": total_wall_s,
        "stop_reason": stop_reason,
        "final_feasible": bool(sol.feasible),
        "final_obj": float(sol.total_cost) if sol.feasible else None,
        "time_to_first_feasible_s": stats.get("time_to_first_feasible_s"),
        "time_to_last_improvement_s": stats.get("time_to_last_improvement_s"),
        "num_improvements": int(stats.get("num_improvements", len(stats.get("improvement_history", [])))),
        "source": instance_json,
        "objective": sol.total_cost,
        "hubs": list(sol.open_hubs),
        "assign": {str(i): int(h) for i, h in sol.assignment.items()},
        "tours": {str(h): [list(r) for r in sol.routes.get(h, [])] for h in sol.open_hubs},
        "costs": {
            "route": sol.route_cost,
            "assignment": sol.assignment_cost,
            "interhub": sol.interhub_cost,
            "vehicle": sol.vehicle_cost,
            "hub": sol.hub_cost,
        },
        "benchmark_params": {
            "alpha": inst.alpha,
            "q": inst.q,
            "F_vehicle": float(getattr(inst, "vehicle_fixed_cost_param", 0.0)),
            "Lambda": list(inst.hub_capacity),
            "Fhub": list(inst.hub_fixed_cost),
        },
        "vns_settings": settings,
        "vns_stats": stats,
        "feasible": sol.feasible,
        "violations": list(sol.violations or []),
        "perf": perf,
        "improvement_history": stats.get("improvement_history", []),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified VNS for the integration HLRP benchmark")
    p.add_argument("--instance_json", type=str, required=True)
    p.add_argument("--time_limit", type=float, default=60.0)
    p.add_argument("--kmax", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max_open", type=int, default=0, help="0 means automatic")
    p.add_argument("--out", type=str, default="")
    return p


def main() -> None:
    args = build_parser().parse_args()
    inst = load_instance(args.instance_json)
    solver = UnifiedVNS(inst, seed=args.seed, max_open=(None if args.max_open <= 0 else args.max_open))
    sol, stats = solver.solve(kmax=max(1, args.kmax), time_limit=max(1.0, args.time_limit))
    out = build_output(
        args.instance_json,
        inst,
        sol,
        {
            "seed": args.seed,
            "time_limit": float(args.time_limit),
            "kmax": int(args.kmax),
            "max_open": solver.max_open,
        },
        stats,
    )
    txt = json.dumps(out, ensure_ascii=False, indent=2)
    print(txt)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(txt)


if __name__ == "__main__":
    main()
