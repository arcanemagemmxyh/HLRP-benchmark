# -*- coding: utf-8 -*-
"""
Unified ALNS for the integration benchmark.

This version reuses the broad SAHLRP-ALNS structure from the existing codebase,
but adapts it to the unified benchmark JSON produced by
src/generator/generate_unified_hlrp_dataset.py and evaluated by
src/evaluator/hlrp_instance.py.

Main benchmark assumptions
- single allocation
- variable number of open hubs
- hub capacity only
- route cardinality bound q
- local closed tours
- hub fixed cost + local routing cost + assignment cost + inter-hub cost + vehicle fixed cost

Usage example
python src/algorithms/alns.py --instance_json data/main_core/loose/ap20_loose_q5.json --time_limit 60 --seed 11 --out scratch/ap20_loose_alns.json
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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_COMMON_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "evaluator"))
if _COMMON_DIR not in sys.path:
    sys.path.insert(0, _COMMON_DIR)

from hlrp_instance import load_instance, HLRPInstance  # type: ignore


@dataclass
class ALNSSolution:
    open_hubs: List[int]
    assignment: Dict[int, int]
    routes: Dict[int, List[List[int]]]
    total_cost: float = float("inf")
    hub_cost: float = 0.0
    route_cost: float = 0.0
    assignment_cost: float = 0.0
    interhub_cost: float = 0.0
    vehicle_cost: float = 0.0
    hub_violation: float = 0.0
    route_violation: float = 0.0
    penalized_cost: float = float("inf")
    feasible: bool = False
    violations: Optional[List[str]] = None


class UnifiedALNS:
    def __init__(
        self,
        inst: HLRPInstance,
        seed: int = 1,
        rho: float = 0.30,
        use_adaptive_weights: bool = True,
        reheat_gap: int = 1200,
        reheat_factor: float = 0.25,
        restart_after_reheats: int = 2,
        min_open_hubs: int = 0,
    ):
        self.inst = inst
        self.seed = int(seed)
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.rho = max(0.05, min(0.80, float(rho)))
        self.n = inst.n
        self.nodes = list(range(self.n))
        self.assignment_weight = float(getattr(inst, "assignment_weight", 0.0))
        mean_tp = sum(inst.outbound_flow[i] + inst.inbound_flow[i] for i in self.nodes) / max(1, self.n)
        if mean_tp <= 1e-12:
            mean_tp = 1.0
        self.node_tp_norm = [(inst.outbound_flow[i] + inst.inbound_flow[i]) / mean_tp for i in self.nodes]

        self.mu = 1e4
        self.phi = 1e4
        self.mu_min = 1.0
        self.phi_min = 1.0
        self.mu_max = 1e8
        self.phi_max = 1e8

        self.subproblem_weights = {"hub": 1.0, "alloc": 1.0, "route": 1.0}
        self.operator_weights = {
            "hub": {"swap_hub": 1.0, "open_hub": 1.0, "close_hub": 1.0},
            "alloc": {"random_reassign": 1.0, "far_reassign": 1.0},
            "route": {"RR": 1.0, "WCR": 1.0, "SR": 1.0, "RRR": 1.0},
        }
        self.repair_weights = {"GI": 1.0, "RI": 1.0}
        self.segment = 50
        self.eta = 0.9
        self.min_weight = 0.05
        self.score_global = 8.0
        self.score_improve = 5.0
        self.score_accept = 1.0
        self.score_feasible = 6.0
        self.use_adaptive_weights = bool(use_adaptive_weights)
        self.reheat_gap = max(0, int(reheat_gap))
        self.reheat_factor = max(0.05, min(1.0, float(reheat_factor)))
        self.restart_after_reheats = max(0, int(restart_after_reheats))
        self.min_open_hubs = max(0, int(min_open_hubs))
        self.stats = self._empty_stats()

    def _empty_stats(self):
        return {
            "subproblem": {k: {"score": 0.0, "count": 0} for k in self.subproblem_weights},
            "operator": {
                sp: {op: {"score": 0.0, "count": 0} for op in ops}
                for sp, ops in self.operator_weights.items()
            },
            "repair": {k: {"score": 0.0, "count": 0} for k in self.repair_weights},
        }

    def clone(self, sol: ALNSSolution) -> ALNSSolution:
        return copy.deepcopy(sol)

    def weighted_choice(self, weights: Dict[str, float]) -> str:
        keys = list(weights.keys())
        vals = np.array([max(self.min_weight, float(weights[k])) for k in keys], dtype=float)
        probs = vals / vals.sum()
        return keys[int(self.np_rng.choice(len(keys), p=probs))]

    def removal_count(self) -> int:
        return max(1, int(math.ceil(self.rho * self.n)))

    def hub_load(self, sol: ALNSSolution, hid: int) -> float:
        return sum(
            self.inst.outbound_flow[i] + self.inst.inbound_flow[i]
            for i, h in sol.assignment.items()
            if h == hid and i not in sol.open_hubs
        )

    def route_cost(self, hid: int, route: List[int]) -> float:
        return self.inst.route_cost(hid, route)

    def assignment_cost_of(self, customer: int, hub: int) -> float:
        if customer == hub:
            return 0.0
        if hasattr(self.inst, "assignment_cost_of"):
            return float(self.inst.assignment_cost_of(customer, hub))
        return float(self.assignment_weight * self.node_tp_norm[customer] * self.inst.distance[customer][hub])

    def route_marginal_remove(self, hid: int, route: List[int], idx: int) -> float:
        cid = route[idx]
        prev_node = None if idx == 0 else route[idx - 1]
        next_node = None if idx == len(route) - 1 else route[idx + 1]
        before = self.inst.distance[hid][cid] if prev_node is None else self.inst.distance[prev_node][cid]
        before += self.inst.distance[cid][hid] if next_node is None else self.inst.distance[cid][next_node]
        if prev_node is None and next_node is None:
            after = 0.0
        elif prev_node is None:
            after = self.inst.distance[hid][next_node]
        elif next_node is None:
            after = self.inst.distance[prev_node][hid]
        else:
            after = self.inst.distance[prev_node][next_node]
        return float(before - after)

    def build_routes_from_assignment(self, open_hubs: List[int], assignment: Dict[int, int]) -> Dict[int, List[List[int]]]:
        routes: Dict[int, List[List[int]]] = {h: [] for h in open_hubs}
        by_hub: Dict[int, List[int]] = {h: [] for h in open_hubs}
        for cid, hid in assignment.items():
            if cid in open_hubs:
                continue
            if hid in by_hub:
                by_hub[hid].append(cid)

        coords = self.inst.nodes
        for hid in open_hubs:
            custs = by_hub.get(hid, [])[:]
            if not custs:
                continue
            hx = float(coords[hid].get("x", 0.0)) if hid < len(coords) else 0.0
            hy = float(coords[hid].get("y", 0.0)) if hid < len(coords) else 0.0
            custs.sort(key=lambda c: math.atan2(float(coords[c].get("y", 0.0)) - hy, float(coords[c].get("x", 0.0)) - hx))
            chunks = [custs[i:i + self.inst.q] for i in range(0, len(custs), self.inst.q)]
            routes[hid] = [self.improve_route(hid, r) for r in chunks if r]
        return routes

    def canonicalize_solution(self, sol: ALNSSolution) -> None:
        open_hubs = sorted(set(int(h) for h in sol.open_hubs))
        if not open_hubs:
            open_hubs = [0]
        customer_set = {i for i in self.nodes if i not in open_hubs}

        assignment: Dict[int, int] = {}
        for h in open_hubs:
            assignment[h] = h

        # keep only customer assignments to currently open hubs
        for cid, hid in list(sol.assignment.items()):
            cid = int(cid)
            hid = int(hid)
            if cid in open_hubs:
                continue
            if cid not in customer_set:
                continue
            if hid not in open_hubs:
                continue
            assignment[cid] = hid

        # assign missing customers greedily by assignment cost with a light capacity preference
        current_load = {h: 0.0 for h in open_hubs}
        for cid, hid in assignment.items():
            if cid in open_hubs:
                continue
            current_load[hid] += self.inst.outbound_flow[cid] + self.inst.inbound_flow[cid]

        missing = [c for c in customer_set if c not in assignment]
        for cid in missing:
            load = self.inst.outbound_flow[cid] + self.inst.inbound_flow[cid]
            best_h = None
            best_score = float("inf")
            for hid in open_hubs:
                over = max(0.0, current_load[hid] + load - self.inst.hub_capacity[hid])
                score = self.assignment_cost_of(cid, hid) + 1e6 * over
                if score < best_score:
                    best_score = score
                    best_h = hid
            assignment[cid] = int(best_h)
            current_load[int(best_h)] += load

        open_hubs, assignment = self.capacity_repair_assignment(open_hubs, assignment)

        # preserve valid route order where possible, but rebuild strictly from assignments
        assigned_by_hub: Dict[int, List[int]] = {h: [] for h in open_hubs}
        for cid, hid in assignment.items():
            if cid in open_hubs:
                continue
            assigned_by_hub[hid].append(cid)

        cleaned_routes: Dict[int, List[List[int]]] = {h: [] for h in open_hubs}
        for hid in open_hubs:
            assigned = set(assigned_by_hub[hid])
            ordered: List[int] = []
            seen = set()
            for route in sol.routes.get(hid, []):
                for c in route:
                    c = int(c)
                    if c in assigned and c not in seen and c not in open_hubs:
                        ordered.append(c)
                        seen.add(c)
            remaining = [c for c in assigned_by_hub[hid] if c not in seen]
            if remaining:
                hx = float(self.inst.nodes[hid].get("x", 0.0)) if hid < len(self.inst.nodes) else 0.0
                hy = float(self.inst.nodes[hid].get("y", 0.0)) if hid < len(self.inst.nodes) else 0.0
                remaining.sort(key=lambda c: math.atan2(float(self.inst.nodes[c].get("y", 0.0)) - hy, float(self.inst.nodes[c].get("x", 0.0)) - hx))
                ordered.extend(remaining)
            for i in range(0, len(ordered), self.inst.q):
                chunk = ordered[i:i + self.inst.q]
                if chunk:
                    cleaned_routes[hid].append(self.improve_route(hid, chunk))

        sol.open_hubs = open_hubs
        sol.assignment = assignment
        sol.routes = cleaned_routes

    def capacity_repair_assignment(self, open_hubs: List[int], assignment: Dict[int, int]) -> Tuple[List[int], Dict[int, int]]:
        open_hubs = sorted(set(int(h) for h in open_hubs))
        assignment = {int(k): int(v) for k, v in assignment.items()}
        for h in open_hubs:
            assignment[h] = h

        def load_of(c: int) -> float:
            return float(self.inst.outbound_flow[c] + self.inst.inbound_flow[c])

        def compute_loads() -> Dict[int, float]:
            loads = {h: 0.0 for h in open_hubs}
            for c, h in assignment.items():
                if c in open_hubs:
                    continue
                if h in loads:
                    loads[h] += load_of(c)
            return loads

        for _ in range(max(1, 2 * self.n)):
            loads = compute_loads()
            overloaded = [h for h in open_hubs if loads[h] - self.inst.hub_capacity[h] > 1e-9]
            if not overloaded:
                break

            source_h = max(overloaded, key=lambda h: loads[h] - self.inst.hub_capacity[h])
            customers = [c for c, h in assignment.items() if h == source_h and c not in open_hubs]
            if not customers:
                break
            customers.sort(key=load_of, reverse=True)

            moved = False
            for c in customers:
                demand = load_of(c)
                best_target = None
                best_score = float("inf")

                for h in open_hubs:
                    if h == source_h:
                        continue
                    if loads[h] + demand <= self.inst.hub_capacity[h] + 1e-9:
                        score = self.assignment_cost_of(c, h)
                        if score < best_score:
                            best_score = score
                            best_target = h

                if best_target is not None:
                    assignment[c] = int(best_target)
                    moved = True
                    break

                closed = [h for h in self.nodes if h not in open_hubs]
                feasible_closed = [h for h in closed if h == c or demand <= self.inst.hub_capacity[h] + 1e-9]
                if not feasible_closed:
                    continue

                def open_score(h: int) -> float:
                    assign_cost = 0.0 if h == c else self.assignment_cost_of(c, h)
                    return float(self.inst.hub_fixed_cost[h] + assign_cost)

                new_h = min(feasible_closed, key=open_score)
                open_hubs.append(int(new_h))
                open_hubs = sorted(set(open_hubs))
                assignment[new_h] = int(new_h)
                if new_h != c:
                    assignment[c] = int(new_h)
                moved = True
                break

            if not moved:
                break

        open_set = set(open_hubs)
        repaired = {h: h for h in open_hubs}
        for c, h in assignment.items():
            c = int(c)
            h = int(h)
            if c in open_set:
                repaired[c] = c
            elif h in open_set:
                repaired[c] = h
        return sorted(open_hubs), repaired

    def improve_route_2opt(self, hid: int, route: List[int]) -> List[int]:
        if len(route) <= 2:
            return route[:]
        best = route[:]
        best_cost = self.route_cost(hid, best)
        improved = True
        while improved:
            improved = False
            for i in range(len(best) - 1):
                for j in range(i + 1, len(best)):
                    cand = best[:i] + list(reversed(best[i:j + 1])) + best[j + 1:]
                    c = self.route_cost(hid, cand)
                    if c + 1e-9 < best_cost:
                        best = cand
                        best_cost = c
                        improved = True
                        break
                if improved:
                    break
        return best

    def improve_route_relocate(self, hid: int, route: List[int]) -> List[int]:
        if len(route) <= 2:
            return route[:]
        best = route[:]
        best_cost = self.route_cost(hid, best)
        improved = True
        while improved:
            improved = False
            for i in range(len(best)):
                for j in range(len(best)):
                    if i == j:
                        continue
                    cand = best[:]
                    node = cand.pop(i)
                    cand.insert(j, node)
                    c = self.route_cost(hid, cand)
                    if c + 1e-9 < best_cost:
                        best = cand
                        best_cost = c
                        improved = True
                        break
                if improved:
                    break
        return best

    def improve_route(self, hid: int, route: List[int]) -> List[int]:
        best = self.improve_route_2opt(hid, route)
        best = self.improve_route_relocate(hid, best)
        best = self.improve_route_2opt(hid, best)
        return best

    def evaluate(self, sol: ALNSSolution) -> ALNSSolution:
        self.canonicalize_solution(sol)
        open_hubs = sorted(set(int(h) for h in sol.open_hubs))
        customer_set = {i for i in self.nodes if i not in open_hubs}
        assignment = {int(k): int(v) for k, v in sol.assignment.items()}
        for h in open_hubs:
            assignment[h] = h

        route_cost = 0.0
        vehicle_cost = 0.0
        route_violation = 0.0
        seen = set()
        cleaned_routes: Dict[int, List[List[int]]] = {h: [] for h in open_hubs}

        for h in open_hubs:
            for route in sol.routes.get(h, []):
                if not route:
                    continue
                r = [int(c) for c in route]
                if len(r) > self.inst.q:
                    route_violation += float(len(r) - self.inst.q)
                local_seen = set()
                bad = False
                for c in r:
                    if c in open_hubs or c not in customer_set:
                        route_violation += 1.0
                        bad = True
                    if c in local_seen or c in seen:
                        route_violation += 1.0
                        bad = True
                    local_seen.add(c)
                    seen.add(c)
                    if assignment.get(c) != h:
                        route_violation += 1.0
                        bad = True
                cleaned_routes[h].append(r)
                if not bad:
                    route_cost += self.route_cost(h, r)
                    vehicle_cost += self.inst.vehicle_fixed_cost_param
                else:
                    route_cost += self.route_cost(h, r)
                    vehicle_cost += self.inst.vehicle_fixed_cost_param

        for c in customer_set:
            if c not in assignment:
                route_violation += 1.0
            elif assignment[c] not in open_hubs:
                route_violation += 1.0
            elif c not in seen:
                route_violation += 1.0

        hub_loads = {h: 0.0 for h in open_hubs}
        for c in customer_set:
            h = assignment.get(c, None)
            if h is None:
                continue
            hub_loads[h] = hub_loads.get(h, 0.0) + self.inst.outbound_flow[c] + self.inst.inbound_flow[c]
        hub_violation = 0.0
        for h in open_hubs:
            hub_violation += max(0.0, hub_loads.get(h, 0.0) - self.inst.hub_capacity[h])

        hub_cost = sum(self.inst.hub_fixed_cost[h] for h in open_hubs)
        assignment_cost = 0.0
        for c in customer_set:
            h = assignment.get(c, None)
            if h is None:
                continue
            assignment_cost += self.assignment_cost_of(c, h)

        interhub_cost = 0.0
        for i in self.nodes:
            hi = assignment.get(i, i)
            for j in self.nodes:
                w = self.inst.flow[i][j]
                if w == 0.0:
                    continue
                hj = assignment.get(j, j)
                interhub_cost += self.inst.alpha * w * self.inst.distance[hi][hj]

        total_cost = hub_cost + route_cost + assignment_cost + interhub_cost + vehicle_cost
        feasible = (hub_violation <= 1e-9 and route_violation <= 1e-9 and len(open_hubs) > 0)
        penalized = total_cost + self.mu * route_violation + self.phi * hub_violation

        sol.open_hubs = open_hubs
        sol.routes = cleaned_routes
        sol.assignment = assignment
        sol.total_cost = float(total_cost)
        sol.hub_cost = float(hub_cost)
        sol.route_cost = float(route_cost)
        sol.assignment_cost = float(assignment_cost)
        sol.interhub_cost = float(interhub_cost)
        sol.vehicle_cost = float(vehicle_cost)
        sol.hub_violation = float(hub_violation)
        sol.route_violation = float(route_violation)
        sol.penalized_cost = float(penalized)
        sol.feasible = bool(feasible)
        sol.violations = [] if feasible else [f"hub_violation={hub_violation}", f"route_violation={route_violation}"]
        return sol

    def seed_hub_rank(self) -> List[int]:
        scores = []
        mean_tp = sum(self.inst.outbound_flow[i] + self.inst.inbound_flow[i] for i in self.nodes) / max(1, self.n)
        if mean_tp <= 1e-12:
            mean_tp = 1.0
        for h in self.nodes:
            assign_proxy = 0.0
            for i in self.nodes:
                tp = self.inst.outbound_flow[i] + self.inst.inbound_flow[i]
                assign_proxy += (tp / mean_tp) * self.inst.distance[i][h]
            score = self.inst.hub_fixed_cost[h] + self.assignment_weight * assign_proxy
            scores.append((score, h))
        scores.sort()
        return [h for _, h in scores]

    def initial_solution(self) -> ALNSSolution:
        ranked = self.seed_hub_rank()
        open_hubs = [ranked[0]]
        assignment: Dict[int, int] = {open_hubs[0]: open_hubs[0]}
        remaining_capacity = {h: self.inst.hub_capacity[h] for h in open_hubs}

        customers = [i for i in self.nodes if i not in open_hubs]
        customers.sort(key=lambda i: self.inst.outbound_flow[i] + self.inst.inbound_flow[i], reverse=True)

        for cid in customers:
            load = self.inst.outbound_flow[cid] + self.inst.inbound_flow[cid]
            feasible_open = [h for h in open_hubs if remaining_capacity[h] + 1e-9 >= load]
            if feasible_open:
                best_h = min(feasible_open, key=lambda h: self.assignment_cost_of(cid, h))
                assignment[cid] = best_h
                remaining_capacity[best_h] -= load
                continue

            closed = [h for h in ranked if h not in open_hubs]
            if not closed:
                best_h = min(open_hubs, key=lambda h: self.assignment_cost_of(cid, h))
                assignment[cid] = best_h
                continue
            new_h = closed[0]
            open_hubs.append(new_h)
            assignment[new_h] = new_h
            remaining_capacity[new_h] = self.inst.hub_capacity[new_h]
            assignment[cid] = new_h
            remaining_capacity[new_h] -= load

        routes = self.build_routes_from_assignment(open_hubs, assignment)
        sol = ALNSSolution(open_hubs=sorted(open_hubs), assignment=assignment, routes=routes)
        return self.evaluate(sol)

    def remove_customers(self, sol: ALNSSolution, removed: List[int]) -> None:
        rem = set(removed)
        for cid in removed:
            sol.assignment.pop(cid, None)
        for hid in list(sol.routes.keys()):
            new_routes = []
            for route in sol.routes.get(hid, []):
                nr = [c for c in route if c not in rem]
                if nr:
                    new_routes.append(nr)
            sol.routes[hid] = new_routes

    def insertion_delta(self, sol: ALNSSolution, cid: int, hid: int) -> Tuple[float, Optional[Tuple[int, int]]]:
        routes = sol.routes.setdefault(hid, [])
        best_delta = float("inf")
        best_pos = None
        base_assign = self.assignment_cost_of(cid, hid)
        base_inter = 0.0
        for j, hj in sol.assignment.items():
            if j == cid:
                continue
            base_inter += self.inst.alpha * self.inst.flow[cid][j] * self.inst.distance[hid][hj]
            base_inter += self.inst.alpha * self.inst.flow[j][cid] * self.inst.distance[hj][hid]
        load = self.inst.outbound_flow[cid] + self.inst.inbound_flow[cid]
        over = max(0.0, self.hub_load(sol, hid) + load - self.inst.hub_capacity[hid])
        for r_idx, route in enumerate(routes):
            if len(route) >= self.inst.q:
                continue
            old = self.route_cost(hid, route)
            for pos in range(len(route) + 1):
                cand = route[:pos] + [cid] + route[pos:]
                delta = self.route_cost(hid, cand) - old + base_assign + base_inter + self.phi * over
                if delta < best_delta:
                    best_delta = delta
                    best_pos = (r_idx, pos)
        new_route = [cid]
        new_score = self.route_cost(hid, new_route) + base_assign + base_inter + self.phi * over + self.inst.vehicle_fixed_cost_param
        if new_score < best_delta:
            return new_score, (len(routes), 0)
        return best_delta, best_pos

    def insertion_options(self, sol: ALNSSolution, cid: int, allowed_hubs: Optional[List[int]] = None) -> List[Tuple[float, int]]:
        hubs = allowed_hubs if allowed_hubs is not None else sol.open_hubs
        scored = []
        for hid in hubs:
            delta, _ = self.insertion_delta(sol, cid, hid)
            scored.append((delta, hid))
        scored.sort(key=lambda x: x[0])
        return scored

    def insert_customer(self, sol: ALNSSolution, cid: int, hid: int) -> None:
        _, pos = self.insertion_delta(sol, cid, hid)
        routes = sol.routes.setdefault(hid, [])
        if pos is None:
            routes.append([cid])
        else:
            r_idx, ins = pos
            if r_idx == len(routes):
                routes.append([cid])
            else:
                routes[r_idx] = routes[r_idx][:ins] + [cid] + routes[r_idx][ins:]
        sol.assignment[cid] = hid

    def greedy_insertion(self, sol: ALNSSolution, customers: List[int], restricted_hubs: Optional[Dict[int, List[int]]] = None) -> None:
        for cid in customers:
            allowed = None if restricted_hubs is None else restricted_hubs.get(cid)
            opts = self.insertion_options(sol, cid, allowed)
            best_h = opts[0][1]
            self.insert_customer(sol, cid, best_h)

    def regret_insertion(self, sol: ALNSSolution, customers: List[int], restricted_hubs: Optional[Dict[int, List[int]]] = None) -> None:
        pending = customers[:]
        while pending:
            best_c = pending[0]
            best_h = sol.open_hubs[0]
            best_regret = -1.0
            for cid in pending:
                allowed = None if restricted_hubs is None else restricted_hubs.get(cid)
                scored = self.insertion_options(sol, cid, allowed)
                regret = scored[1][0] - scored[0][0] if len(scored) >= 2 else abs(scored[0][0])
                if regret > best_regret:
                    best_regret = regret
                    best_c = cid
                    best_h = scored[0][1]
            self.insert_customer(sol, best_c, best_h)
            pending.remove(best_c)

    def post_repair_routes(self, sol: ALNSSolution) -> None:
        self.canonicalize_solution(sol)

    def repair(self, sol: ALNSSolution, removed: List[int], repair_name: str, restricted_hubs: Optional[Dict[int, List[int]]] = None) -> None:
        if repair_name == "RI":
            self.regret_insertion(sol, removed, restricted_hubs)
        else:
            self.greedy_insertion(sol, removed, restricted_hubs)
        missing = [i for i in self.nodes if i not in sol.open_hubs and i not in sol.assignment]
        if missing:
            self.greedy_insertion(sol, missing, None)
        self.post_repair_routes(sol)

    def choose_closed_hub(self, sol: ALNSSolution) -> Optional[int]:
        closed = [h for h in self.nodes if h not in sol.open_hubs]
        if not closed:
            return None
        return min(closed, key=lambda h: self.inst.hub_fixed_cost[h])

    def op_swap_hub(self, sol: ALNSSolution):
        cand = self.clone(sol)
        if not cand.open_hubs:
            return cand, [], None
        closed = [h for h in self.nodes if h not in cand.open_hubs]
        if not closed:
            return cand, [], None
        remove_h = self.rng.choice(cand.open_hubs)
        add_h = self.rng.choice(closed)
        removed = [cid for cid, hid in cand.assignment.items() if hid == remove_h and cid != remove_h]
        removed.append(remove_h)  # closed hub becomes a customer again
        cand.open_hubs = [h for h in cand.open_hubs if h != remove_h] + [add_h]
        cand.routes.pop(remove_h, None)
        cand.routes.setdefault(add_h, [])
        cand.assignment.pop(remove_h, None)
        cand.assignment[add_h] = add_h
        self.remove_customers(cand, removed + [add_h])  # newly opened hub must disappear from old customer routes
        restricted = {cid: cand.open_hubs[:] for cid in removed}
        return cand, removed, restricted

    def op_open_hub(self, sol: ALNSSolution):
        cand = self.clone(sol)
        add_h = self.choose_closed_hub(cand)
        if add_h is None:
            return cand, [], None
        cand.open_hubs.append(add_h)
        cand.open_hubs = sorted(set(cand.open_hubs))
        cand.routes.setdefault(add_h, [])
        cand.assignment[add_h] = add_h
        self.remove_customers(cand, [add_h])  # remove newly opened hub from any old customer route
        scored = []
        for cid, hid in list(cand.assignment.items()):
            if cid in cand.open_hubs:
                continue
            old = self.assignment_cost_of(cid, hid)
            new = self.assignment_cost_of(cid, add_h)
            scored.append((old - new, cid))
        scored.sort(reverse=True)
        q = max(1, min(self.removal_count(), max(1, len(scored) // 2)))
        removed = [cid for _, cid in scored[:q]]
        self.remove_customers(cand, removed)
        restricted = {cid: cand.open_hubs[:] for cid in removed}
        return cand, removed, restricted

    def op_close_hub(self, sol: ALNSSolution):
        cand = self.clone(sol)
        if len(cand.open_hubs) <= 1 or len(cand.open_hubs) <= self.min_open_hubs:
            return cand, [], None
        util = []
        for hid in cand.open_hubs:
            load = self.hub_load(cand, hid)
            util.append((load / max(1e-9, self.inst.hub_capacity[hid]), hid))
        util.sort()
        remove_h = util[0][1]
        removed = [cid for cid, hid in cand.assignment.items() if hid == remove_h and cid != remove_h]
        removed.append(remove_h)  # closed hub becomes a customer again
        cand.open_hubs = [h for h in cand.open_hubs if h != remove_h]
        cand.routes.pop(remove_h, None)
        cand.assignment.pop(remove_h, None)
        self.remove_customers(cand, removed)
        restricted = {cid: cand.open_hubs[:] for cid in removed}
        return cand, removed, restricted

    def op_random_reassign(self, sol: ALNSSolution):
        cand = self.clone(sol)
        movable = [i for i in self.nodes if i not in cand.open_hubs]
        q = min(self.removal_count(), len(movable))
        removed = self.rng.sample(movable, q)
        self.remove_customers(cand, removed)
        restricted = {cid: cand.open_hubs[:] for cid in removed}
        return cand, removed, restricted

    def op_far_reassign(self, sol: ALNSSolution):
        cand = self.clone(sol)
        scored = []
        for cid, hid in cand.assignment.items():
            if cid in cand.open_hubs:
                continue
            scored.append((self.assignment_cost_of(cid, hid), cid))
        scored.sort(reverse=True)
        q = min(self.removal_count(), len(scored))
        removed = [cid for _, cid in scored[:q]]
        self.remove_customers(cand, removed)
        restricted = {cid: cand.open_hubs[:] for cid in removed}
        return cand, removed, restricted

    def op_RR(self, sol: ALNSSolution):
        cand = self.clone(sol)
        movable = [i for i in self.nodes if i not in cand.open_hubs]
        q = min(self.removal_count(), len(movable))
        removed = self.rng.sample(movable, q)
        self.remove_customers(cand, removed)
        restricted = {cid: [sol.assignment.get(cid, self.rng.choice(sol.open_hubs))] for cid in removed}
        return cand, removed, restricted

    def op_WCR(self, sol: ALNSSolution):
        cand = self.clone(sol)
        scored = []
        for hid in cand.open_hubs:
            for route in cand.routes.get(hid, []):
                for idx, cid in enumerate(route):
                    scored.append((self.route_marginal_remove(hid, route, idx), cid))
        scored.sort(reverse=True)
        q = min(self.removal_count(), len(scored))
        removed = [cid for _, cid in scored[:q]]
        self.remove_customers(cand, removed)
        restricted = {cid: [sol.assignment.get(cid, self.rng.choice(sol.open_hubs))] for cid in removed}
        return cand, removed, restricted

    def op_SR(self, sol: ALNSSolution):
        cand = self.clone(sol)
        movable = [i for i in self.nodes if i not in cand.open_hubs]
        if not movable:
            return cand, [], None
        seed = self.rng.choice(movable)
        sx = float(self.inst.nodes[seed].get("x", 0.0)) if seed < len(self.inst.nodes) else 0.0
        sy = float(self.inst.nodes[seed].get("y", 0.0)) if seed < len(self.inst.nodes) else 0.0
        sh = sol.assignment.get(seed, self.rng.choice(sol.open_hubs))
        scored = []
        for cid in movable:
            x = float(self.inst.nodes[cid].get("x", 0.0)) if cid < len(self.inst.nodes) else 0.0
            y = float(self.inst.nodes[cid].get("y", 0.0)) if cid < len(self.inst.nodes) else 0.0
            rel = math.hypot(x - sx, y - sy)
            if sol.assignment.get(cid) == sh:
                rel *= 0.8
            scored.append((rel, cid))
        scored.sort(key=lambda t: t[0])
        q = min(self.removal_count(), len(scored))
        removed = [cid for _, cid in scored[:q]]
        self.remove_customers(cand, removed)
        restricted = {cid: [sol.assignment.get(cid, self.rng.choice(sol.open_hubs))] for cid in removed}
        return cand, removed, restricted

    def op_RRR(self, sol: ALNSSolution):
        cand = self.clone(sol)
        all_routes = []
        for hid in cand.open_hubs:
            for ridx, route in enumerate(cand.routes.get(hid, [])):
                all_routes.append((hid, ridx, route[:]))
        if not all_routes:
            return cand, [], None
        hid, ridx, route = self.rng.choice(all_routes)
        removed = route[:]
        cand.routes[hid].pop(ridx)
        self.remove_customers(cand, removed)
        restricted = {cid: [hid] for cid in removed}
        return cand, removed, restricted

    def apply_operator(self, sol: ALNSSolution, subproblem: str, op_name: str):
        if subproblem == "hub":
            if op_name == "swap_hub":
                return self.op_swap_hub(sol)
            if op_name == "open_hub":
                return self.op_open_hub(sol)
            return self.op_close_hub(sol)
        if subproblem == "alloc":
            return self.op_far_reassign(sol) if op_name == "far_reassign" else self.op_random_reassign(sol)
        if op_name == "RR":
            return self.op_RR(sol)
        if op_name == "WCR":
            return self.op_WCR(sol)
        if op_name == "SR":
            return self.op_SR(sol)
        return self.op_RRR(sol)

    def record_score(self, subproblem: str, op_name: str, repair_name: str, score: float) -> None:
        self.stats["subproblem"][subproblem]["score"] += score
        self.stats["subproblem"][subproblem]["count"] += 1
        self.stats["operator"][subproblem][op_name]["score"] += score
        self.stats["operator"][subproblem][op_name]["count"] += 1
        self.stats["repair"][repair_name]["score"] += score
        self.stats["repair"][repair_name]["count"] += 1

    def _renormalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        vals = {k: max(self.min_weight, float(v)) for k, v in weights.items()}
        mean_val = sum(vals.values()) / max(1, len(vals))
        if mean_val <= 1e-12:
            return {k: 1.0 for k in weights}
        scaled = {k: max(self.min_weight, v / mean_val) for k, v in vals.items()}
        mean_scaled = sum(scaled.values()) / max(1, len(scaled))
        return {k: max(self.min_weight, v / max(1e-12, mean_scaled)) for k, v in scaled.items()}

    def flush_weight_updates(self) -> None:
        if not self.use_adaptive_weights:
            self.stats = self._empty_stats()
            return
        for k, v in self.subproblem_weights.items():
            st = self.stats["subproblem"][k]
            avg = st["score"] / st["count"] if st["count"] > 0 else v
            self.subproblem_weights[k] = max(self.min_weight, self.eta * v + (1.0 - self.eta) * avg)
        self.subproblem_weights = self._renormalize_weights(self.subproblem_weights)
        for sp, ops in self.operator_weights.items():
            updated = {}
            for op, val in ops.items():
                st = self.stats["operator"][sp][op]
                avg = st["score"] / st["count"] if st["count"] > 0 else val
                updated[op] = max(self.min_weight, self.eta * val + (1.0 - self.eta) * avg)
            self.operator_weights[sp] = self._renormalize_weights(updated)
        self.repair_weights = self._renormalize_weights({
            k: max(
                self.min_weight,
                self.eta * v + (1.0 - self.eta) * (self.stats["repair"][k]["score"] / self.stats["repair"][k]["count"] if self.stats["repair"][k]["count"] > 0 else v),
            )
            for k, v in self.repair_weights.items()
        })
        self.stats = self._empty_stats()

    def soft_restart_from(self, base: ALNSSolution) -> ALNSSolution:
        cand = self.clone(base)
        movable = [i for i in self.nodes if i not in cand.open_hubs]
        if not movable:
            return self.evaluate(cand)
        q = min(max(1, self.removal_count()), len(movable))
        removed = self.rng.sample(movable, q)
        self.remove_customers(cand, removed)
        restricted = {cid: cand.open_hubs[:] for cid in removed}
        repair_name = "RI" if len(removed) >= 3 else "GI"
        self.repair(cand, removed, repair_name, restricted)
        return self.evaluate(cand)

    def update_penalties(self, current: ALNSSolution) -> None:
        if current.route_violation > 1e-9:
            self.mu = min(self.mu_max, self.mu * 1.05)
        else:
            self.mu = max(self.mu_min, self.mu / 1.02)
        if current.hub_violation > 1e-9:
            self.phi = min(self.phi_max, self.phi * 1.05)
        else:
            self.phi = max(self.phi_min, self.phi / 1.02)

    
    def solve(self, iters: int = 3000, init_temp: Optional[float] = None, cooling: float = 0.999, verbose_every: int = 200, time_limit: float = 0.0):
            verbose_every = max(1, int(verbose_every))
            deadline = (time.time() + float(time_limit)) if float(time_limit) > 0 else None
            current = self.initial_solution()
            best_pen = self.clone(current)
            best_feasible = self.clone(current) if current.feasible else None
            if init_temp is None:
                init_temp = max(100.0, 0.2 * max(1.0, current.total_cost))
            temp = init_temp
            history = []
            t0 = time.time()
            reheats = 0
            restarts = 0
            last_progress_iter = 0
            stopped_by_time = False
            improvement_history: List[Dict[str, object]] = []
            first_feasible_obj = None
            best_feasible_obj = None
            time_to_first_feasible = None
            time_to_best_feasible = None
            first_feasible_step = None
            best_feasible_step = None
            if best_feasible is not None:
                first_feasible_obj = float(best_feasible.total_cost)
                best_feasible_obj = float(best_feasible.total_cost)
                time_to_first_feasible = 0.0
                time_to_best_feasible = 0.0
                first_feasible_step = 0
                best_feasible_step = 0
                improvement_history.append({"t_s": 0.0, "step": 0, "obj": float(best_feasible.total_cost)})
    
            actual_iters = 0
            for it in range(1, iters + 1):
                if deadline is not None and time.time() >= deadline:
                    stopped_by_time = True
                    break
                actual_iters = it
                subproblem = self.weighted_choice(self.subproblem_weights)
                op_weights = self.operator_weights[subproblem]
                if subproblem == "hub" and self.min_open_hubs > 0 and len(current.open_hubs) <= self.min_open_hubs:
                    op_weights = {k: v for k, v in self.operator_weights[subproblem].items() if k != "close_hub"}
                    if not op_weights:
                        op_weights = self.operator_weights[subproblem]
                op_name = self.weighted_choice(op_weights)
                repair_name = self.weighted_choice(self.repair_weights)
    
                if deadline is not None and time.time() >= deadline:
                    stopped_by_time = True
                    break
                cand, removed, restricted = self.apply_operator(current, subproblem, op_name)
                if deadline is not None and time.time() >= deadline:
                    stopped_by_time = True
                    break
                if removed:
                    self.repair(cand, removed, repair_name, restricted)
                if deadline is not None and time.time() >= deadline:
                    stopped_by_time = True
                    break
                cand = self.evaluate(cand)
    
                delta = cand.penalized_cost - current.penalized_cost
                accept = False
                if delta <= 1e-9:
                    accept = True
                else:
                    prob = math.exp(-delta / max(1e-9, temp))
                    if self.rng.random() <= prob:
                        accept = True
    
                score = 0.0
                improved_global = False
                if accept:
                    prev = current
                    current = cand
                    score = self.score_accept
                    if current.penalized_cost + 1e-9 < best_pen.penalized_cost:
                        best_pen = self.clone(current)
                        score = self.score_global
                        improved_global = True
                    elif current.penalized_cost + 1e-9 < prev.penalized_cost:
                        score = self.score_improve
                    if current.feasible and (best_feasible is None or current.total_cost + 1e-9 < best_feasible.total_cost):
                        best_feasible = self.clone(current)
                        score = max(score, self.score_feasible)
                        improved_global = True
                        now_t = float(time.time() - t0)
                        if first_feasible_obj is None:
                            first_feasible_obj = float(current.total_cost)
                            time_to_first_feasible = now_t
                            first_feasible_step = it
                        best_feasible_obj = float(current.total_cost)
                        time_to_best_feasible = now_t
                        best_feasible_step = it
                        improvement_history.append({"t_s": now_t, "step": int(it), "obj": float(current.total_cost)})
    
                if improved_global:
                    last_progress_iter = it
    
                self.record_score(subproblem, op_name, repair_name, score)
                if it % self.segment == 0:
                    self.flush_weight_updates()
                self.update_penalties(current)
                temp *= cooling
    
                if self.reheat_gap > 0 and (it - last_progress_iter) >= self.reheat_gap:
                    temp = max(temp, init_temp * self.reheat_factor)
                    reheats += 1
                    last_progress_iter = it
                    base = best_feasible if best_feasible is not None else best_pen
                    print(
                        f"[ALNS-unified] reheat={reheats} at it={it} temp={temp:.2f} "
                        f"best_feas={None if best_feasible is None else round(best_feasible.total_cost, 6)}"
                    )
                    if self.restart_after_reheats > 0 and reheats % self.restart_after_reheats == 0:
                        current = self.soft_restart_from(base)
                        restarts += 1
                        print(
                            f"[ALNS-unified] restart={restarts} at it={it} curr={current.total_cost:.6f} "
                            f"feas={current.feasible} hubs={len(current.open_hubs)}"
                        )
    
                if it % verbose_every == 0 or it == 1 or it == iters:
                    history.append({
                        "iter": it,
                        "temp": temp,
                        "current_total": current.total_cost,
                        "current_penalized": current.penalized_cost,
                        "best_pen_total": best_pen.total_cost,
                        "best_feas_total": None if best_feasible is None else best_feasible.total_cost,
                        "hub_violation": current.hub_violation,
                        "route_violation": current.route_violation,
                        "mu": self.mu,
                        "phi": self.phi,
                        "open_hubs": len(current.open_hubs),
                        "reheats": reheats,
                        "restarts": restarts,
                    })
                    print(
                        f"[ALNS-unified] it={it:5d} temp={temp:.2f} best_pen={best_pen.total_cost:.6f} "
                        f"best_feas={None if best_feasible is None else round(best_feasible.total_cost, 6)} "
                        f"curr={current.total_cost:.6f} hubs={len(current.open_hubs)} "
                        f"hv={current.hub_violation:.3f} rv={current.route_violation:.3f} "
                        f"reh={reheats} rst={restarts}"
                    )
    
            wall = time.time() - t0
            best = best_feasible if best_feasible is not None else best_pen
            time_to_last_improvement = time_to_best_feasible
            last_improvement_step = best_feasible_step
            dataset_type = str(self.inst.instance_name).split("_")[1] if "_" in str(self.inst.instance_name) else None
            stop_reason = "time_limit" if stopped_by_time else "iteration_cap"
            runtime_utilization = None if float(time_limit) <= 0 else float(min(1.0, wall / max(float(time_limit), 1e-12)))
            return {
                "instance": self.inst.instance_name,
                "dataset_type": dataset_type,
                "method": "ALNS",
                "seed": int(self.seed),
                "time_limit_s": float(time_limit) if float(time_limit) > 0 else None,
                "total_wall_s": float(wall),
                "stop_reason": stop_reason,
                "final_feasible": bool(best.feasible),
                "final_obj": float(best.total_cost) if best.feasible else None,
                "source": None,
                "objective": best.total_cost,
                "hubs": list(best.open_hubs),
                "assign": {str(k): int(v) for k, v in best.assignment.items() if k not in best.open_hubs},
                "tours": {str(h): [list(r) for r in best.routes.get(h, [])] for h in best.open_hubs},
                "costs": {
                    "route": best.route_cost,
                    "assignment": best.assignment_cost,
                    "interhub": best.interhub_cost,
                    "vehicle": best.vehicle_cost,
                    "hub": best.hub_cost,
                },
                "perf": {
                    "algorithm": "ALNS",
                    "seed": int(self.seed),
                    "instance": self.inst.instance_name,
                    "step_unit": "alns_iteration",
                    "stopped_by_time": bool(stopped_by_time),
                    "stop_reason": stop_reason,
                    "time_limit_s": float(time_limit) if float(time_limit) > 0 else None,
                    "total_wall_s": float(wall),
                    "runtime_utilization": runtime_utilization,
                    "first_feasible_obj": None if first_feasible_obj is None else float(first_feasible_obj),
                    "best_feasible_obj": None if best_feasible_obj is None else float(best_feasible_obj),
                    "time_to_first_feasible_s": None if time_to_first_feasible is None else float(time_to_first_feasible),
                    "time_to_best_feasible_s": None if time_to_best_feasible is None else float(time_to_best_feasible),
                    "time_to_last_improvement_s": None if time_to_last_improvement is None else float(time_to_last_improvement),
                    "first_feasible_step": first_feasible_step,
                    "best_feasible_step": best_feasible_step,
                    "last_improvement_step": last_improvement_step,
                    "num_improvements": int(len(improvement_history)),
                    "final_feasible": bool(best.feasible),
                },
                "improvement_history": improvement_history,
                "benchmark_params": {
                    "alpha": self.inst.alpha,
                    "q": self.inst.q,
                    "assignment_weight": getattr(self.inst, "assignment_weight", 0.0),
                    "F_vehicle": self.inst.vehicle_fixed_cost_param,
                    "Lambda": list(self.inst.hub_capacity),
                    "Fhub": list(self.inst.hub_fixed_cost),
                },
                "alns_settings": {
                    "iters": iters,
                    "actual_iters": actual_iters,
                    "seed": self.seed,
                    "rho": self.rho,
                    "init_temp": init_temp,
                    "cooling": cooling,
                    "segment": self.segment,
                    "use_adaptive_weights": self.use_adaptive_weights,
                    "reheat_gap": self.reheat_gap,
                    "reheat_factor": self.reheat_factor,
                    "restart_after_reheats": self.restart_after_reheats,
                    "min_open_hubs": self.min_open_hubs,
                    "time_limit": float(time_limit) if float(time_limit) > 0 else None,
                },
                "alns_stats": {
                    "wall_s": wall,
                    "stopped_by_time": bool(stopped_by_time),
                    "reheats": reheats,
                    "restarts": restarts,
                    "final_weights": {
                        "subproblem": self.subproblem_weights,
                        "operator": self.operator_weights,
                        "repair": self.repair_weights,
                    },
                    "history": history,
                },
                "feasible": bool(best.feasible),
                "violations": list(best.violations or []),
            }
    
    
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_json", type=str, required=True)
    parser.add_argument("--iters", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rho", type=float, default=0.30)
    parser.add_argument("--init_temp", type=float, default=-1.0)
    parser.add_argument("--cooling", type=float, default=0.999)
    parser.add_argument("--verbose_every", type=int, default=200)
    parser.add_argument("--use_adaptive_weights", type=int, default=1)
    parser.add_argument("--reheat_gap", type=int, default=1200)
    parser.add_argument("--reheat_factor", type=float, default=0.25)
    parser.add_argument("--restart_after_reheats", type=int, default=2)
    parser.add_argument("--min_open_hubs", type=int, default=0)
    parser.add_argument("--time_limit", type=float, default=0.0)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    inst = load_instance(args.instance_json)
    solver = UnifiedALNS(
        inst=inst,
        seed=args.seed,
        rho=args.rho,
        use_adaptive_weights=bool(args.use_adaptive_weights),
        reheat_gap=args.reheat_gap,
        reheat_factor=args.reheat_factor,
        restart_after_reheats=args.restart_after_reheats,
        min_open_hubs=args.min_open_hubs,
    )
    result = solver.solve(
        iters=args.iters,
        init_temp=None if args.init_temp <= 0 else args.init_temp,
        cooling=args.cooling,
        verbose_every=args.verbose_every,
        time_limit=args.time_limit,
    )
    result["source"] = args.instance_json
    out_path = Path(args.out or f"{inst.instance_name}_alns_unified.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"saved_to={os.path.abspath(out_path)}")
    print(
        f"best_total={result['objective']:.6f} hubs={len(result['hubs'])} "
        f"route={result['costs']['route']:.6f} assignment={result['costs']['assignment']:.6f} "
        f"interhub={result['costs']['interhub']:.6f} hub={result['costs']['hub']:.6f}"
    )


if __name__ == "__main__":
    main()
