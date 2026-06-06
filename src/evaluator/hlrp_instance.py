# -*- coding: utf-8 -*-
r"""
hlrp_instance.py

Unified instance loader and evaluator for the Karimi-based HLRP benchmark JSON
produced by generate_unified_hlrp_dataset_v2.py.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple


@dataclass(frozen=True)
class EvaluationResult:
    feasible: bool
    total_cost: float
    hub_fixed_cost: float
    local_route_cost: float
    assignment_cost: float
    inter_hub_flow_cost: float
    vehicle_fixed_cost: float
    n_open_hubs: int
    n_routes: int
    hub_loads: Dict[int, float]
    route_costs: Dict[int, List[float]]
    violations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feasible": self.feasible,
            "total_cost": self.total_cost,
            "hub_fixed_cost": self.hub_fixed_cost,
            "local_route_cost": self.local_route_cost,
            "assignment_cost": self.assignment_cost,
            "inter_hub_flow_cost": self.inter_hub_flow_cost,
            "vehicle_fixed_cost": self.vehicle_fixed_cost,
            "n_open_hubs": self.n_open_hubs,
            "n_routes": self.n_routes,
            "hub_loads": dict(self.hub_loads),
            "route_costs": {int(k): list(v) for k, v in self.route_costs.items()},
            "violations": list(self.violations),
        }


class HLRPInstance:
    def __init__(self, data: Mapping[str, Any]) -> None:
        self.data = dict(data)
        self.instance_name: str = str(data["instance_name"])
        self.n: int = int(data["summary"]["n_nodes"])
        self.alpha: float = float(data["parameters"]["alpha"])
        self.q: int = int(data["parameters"]["q"])
        self.vehicle_fixed_cost_param: float = float(data["parameters"].get("vehicle_fixed_cost", 0.0))
        self.assignment_weight: float = float(data["parameters"].get("assignment_weight", 0.0))
        self.distance: List[List[float]] = [[float(x) for x in row] for row in data["distance_matrix"]]
        self.flow: List[List[float]] = [[float(x) for x in row] for row in data["flow_matrix"]]
        self.hub_fixed_cost: List[float] = [float(x) for x in data["hub_fixed_cost"]]
        self.hub_capacity: List[float] = [float(x) for x in data["hub_capacity"]]
        self.outbound_flow: List[float] = [float(x) for x in data["outbound_flow"]]
        self.inbound_flow: List[float] = [float(x) for x in data["inbound_flow"]]
        self.node_throughput: List[float] = [float(x) for x in data.get("node_throughput", [])]
        self.node_throughput_normalized: List[float] = [float(x) for x in data.get("node_throughput_normalized", [])]
        self.nodes: List[Dict[str, float]] = list(data.get("nodes", []))
        self._validate_instance()

    def _validate_instance(self) -> None:
        if len(self.distance) != self.n or any(len(row) != self.n for row in self.distance):
            raise ValueError("distance_matrix shape mismatch")
        if len(self.flow) != self.n or any(len(row) != self.n for row in self.flow):
            raise ValueError("flow_matrix shape mismatch")
        if len(self.hub_fixed_cost) != self.n:
            raise ValueError("hub_fixed_cost length mismatch")
        if len(self.hub_capacity) != self.n:
            raise ValueError("hub_capacity length mismatch")
        if len(self.outbound_flow) != self.n or len(self.inbound_flow) != self.n:
            raise ValueError("flow marginal length mismatch")
        if not self.node_throughput:
            self.node_throughput = [self.outbound_flow[i] + self.inbound_flow[i] for i in range(self.n)]
        if not self.node_throughput_normalized:
            mean_tp = sum(self.node_throughput) / max(1, self.n)
            if mean_tp <= 1e-12:
                mean_tp = 1.0
            self.node_throughput_normalized = [x / mean_tp for x in self.node_throughput]
        if len(self.node_throughput) != self.n or len(self.node_throughput_normalized) != self.n:
            raise ValueError("throughput vector length mismatch")

    def route_cost(self, hub: int, route: Sequence[int]) -> float:
        if len(route) == 0:
            return 0.0
        cost = self.distance[hub][route[0]]
        for a, b in zip(route[:-1], route[1:]):
            cost += self.distance[a][b]
        cost += self.distance[route[-1]][hub]
        return float(cost)

    def assignment_cost_of(self, customer: int, hub: int) -> float:
        if customer == hub:
            return 0.0
        return float(self.assignment_weight * self.node_throughput_normalized[customer] * self.distance[customer][hub])

    def _normalize_solution(self, solution: Mapping[str, Any]) -> Tuple[Set[int], Dict[int, int], Dict[int, List[List[int]]]]:
        if "open_hubs" not in solution:
            raise ValueError("solution must contain 'open_hubs'")
        open_hubs = {int(x) for x in solution.get("open_hubs", [])}
        for h in open_hubs:
            self._check_node_id(h, where="open_hubs")

        routes_raw = solution.get("routes", {})
        routes: Dict[int, List[List[int]]] = {}
        if not isinstance(routes_raw, Mapping):
            raise ValueError("solution['routes'] must be a mapping: hub -> list of routes")
        for hk, hub_routes in routes_raw.items():
            h = int(hk)
            routes[h] = []
            for route in hub_routes:
                routes[h].append([int(x) for x in route])

        assignment: Dict[int, int] = {}
        assignment_raw = solution.get("assignment", {}) or {}
        if not isinstance(assignment_raw, Mapping):
            raise ValueError("solution['assignment'] must be a mapping customer -> hub")
        for ck, hv in assignment_raw.items():
            c = int(ck)
            h = int(hv)
            self._check_node_id(c, where="assignment key")
            self._check_node_id(h, where="assignment value")
            assignment[c] = h

        routed_customers: Dict[int, int] = {}
        for h, hub_routes in routes.items():
            self._check_node_id(h, where="routes hub")
            for route in hub_routes:
                for c in route:
                    self._check_node_id(c, where=f"route under hub {h}")
                    if c in routed_customers:
                        raise ValueError(f"customer {c} appears in more than one route")
                    routed_customers[c] = h
                    if c not in assignment:
                        assignment[c] = h
        return open_hubs, assignment, routes

    def _check_node_id(self, node_id: int, where: str) -> None:
        if node_id < 0 or node_id >= self.n:
            raise ValueError(f"invalid node id {node_id} in {where}; valid range is [0, {self.n - 1}]")

    def compute_hub_loads(self, assignment: Mapping[int, int], open_hubs: Iterable[int]) -> Dict[int, float]:
        loads = {int(h): 0.0 for h in open_hubs}
        for i, h in assignment.items():
            if i in loads:
                continue
            loads[h] = loads.get(h, 0.0) + self.outbound_flow[i] + self.inbound_flow[i]
        return loads

    def compute_assignment_cost(self, assignment: Mapping[int, int], open_hubs: Iterable[int]) -> float:
        open_hubs_set = set(int(h) for h in open_hubs)
        cost = 0.0
        for i, h in assignment.items():
            if i in open_hubs_set:
                continue
            cost += self.assignment_cost_of(i, h)
        return float(cost)

    def compute_inter_hub_flow_cost(self, assignment: Mapping[int, int]) -> float:
        cost = 0.0
        for i in range(self.n):
            hi = assignment.get(i, i)
            for j in range(self.n):
                w = self.flow[i][j]
                if w == 0.0:
                    continue
                hj = assignment.get(j, j)
                cost += self.alpha * w * self.distance[hi][hj]
        return float(cost)

    def evaluate_solution(self, solution: Mapping[str, Any], *, raise_on_invalid: bool = False) -> EvaluationResult:
        violations: List[str] = []
        try:
            open_hubs, assignment, routes = self._normalize_solution(solution)
        except Exception as exc:
            if raise_on_invalid:
                raise
            return EvaluationResult(
                feasible=False,
                total_cost=float("inf"),
                hub_fixed_cost=0.0,
                local_route_cost=0.0,
                assignment_cost=0.0,
                inter_hub_flow_cost=0.0,
                vehicle_fixed_cost=0.0,
                n_open_hubs=0,
                n_routes=0,
                hub_loads={},
                route_costs={},
                violations=[str(exc)],
            )

        if not open_hubs:
            violations.append("no hub is opened")

        customers = [i for i in range(self.n) if i not in open_hubs]
        customer_set = set(customers)

        for c in customer_set:
            if c not in assignment:
                violations.append(f"customer {c} has no assignment")
            elif assignment[c] not in open_hubs:
                violations.append(f"customer {c} assigned to closed hub {assignment[c]}")

        for h in open_hubs:
            assignment[h] = h

        seen_in_routes: Set[int] = set()
        route_costs: Dict[int, List[float]] = {int(h): [] for h in open_hubs}
        n_routes = 0

        for h, hub_routes in routes.items():
            if h not in open_hubs:
                violations.append(f"routes provided for closed hub {h}")
            for route in hub_routes:
                n_routes += 1
                if len(route) == 0:
                    violations.append(f"empty route under hub {h}")
                    route_costs.setdefault(h, []).append(0.0)
                    continue
                if len(route) > self.q:
                    violations.append(f"route under hub {h} exceeds q={self.q}: len={len(route)}")
                local_seen: Set[int] = set()
                for c in route:
                    if c in open_hubs:
                        violations.append(f"open hub {c} appears inside a customer route")
                    if c not in customer_set:
                        violations.append(f"node {c} in route under hub {h} is not a customer under current open hubs")
                    if c in local_seen:
                        violations.append(f"customer {c} repeated within one route under hub {h}")
                    local_seen.add(c)
                    if assignment.get(c) != h:
                        violations.append(f"customer {c} appears in route of hub {h} but is assigned to {assignment.get(c)}")
                    if c in seen_in_routes:
                        violations.append(f"customer {c} appears in more than one route")
                    seen_in_routes.add(c)
                route_costs.setdefault(h, []).append(self.route_cost(h, route))

        missing_in_routes = sorted(customer_set - seen_in_routes)
        if missing_in_routes:
            violations.append(f"customers missing from routes: {missing_in_routes}")

        hub_loads = self.compute_hub_loads(assignment, open_hubs)
        for h in open_hubs:
            if hub_loads.get(h, 0.0) - self.hub_capacity[h] > 1e-9:
                violations.append(
                    f"hub {h} capacity violated: load={hub_loads[h]:.6f} > cap={self.hub_capacity[h]:.6f}"
                )

        hub_fixed_cost = sum(self.hub_fixed_cost[h] for h in open_hubs)
        local_route_cost = sum(sum(vals) for vals in route_costs.values())
        assignment_cost = self.compute_assignment_cost(assignment, open_hubs)
        vehicle_fixed_cost = self.vehicle_fixed_cost_param * n_routes
        inter_hub_flow_cost = self.compute_inter_hub_flow_cost(assignment)
        total_cost = hub_fixed_cost + local_route_cost + assignment_cost + vehicle_fixed_cost + inter_hub_flow_cost
        feasible = len(violations) == 0

        return EvaluationResult(
            feasible=feasible,
            total_cost=float(total_cost) if feasible else float("inf"),
            hub_fixed_cost=float(hub_fixed_cost),
            local_route_cost=float(local_route_cost),
            assignment_cost=float(assignment_cost),
            inter_hub_flow_cost=float(inter_hub_flow_cost),
            vehicle_fixed_cost=float(vehicle_fixed_cost),
            n_open_hubs=len(open_hubs),
            n_routes=n_routes,
            hub_loads={int(k): float(v) for k, v in hub_loads.items()},
            route_costs={int(k): [float(x) for x in vals] for k, vals in route_costs.items()},
            violations=violations,
        )

    def check_feasible(self, solution: Mapping[str, Any]) -> Tuple[bool, List[str]]:
        result = self.evaluate_solution(solution)
        return result.feasible, result.violations

    def build_trivial_solution(self, hub: int = 0) -> Dict[str, Any]:
        self._check_node_id(hub, where="build_trivial_solution hub")
        customers = [i for i in range(self.n) if i != hub]
        routes: List[List[int]] = []
        for start in range(0, len(customers), self.q):
            routes.append(customers[start:start + self.q])
        return {
            "open_hubs": [hub],
            "assignment": {str(c): hub for c in customers},
            "routes": {str(hub): routes},
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "instance_name": self.instance_name,
            "n": self.n,
            "alpha": self.alpha,
            "q": self.q,
            "vehicle_fixed_cost": self.vehicle_fixed_cost_param,
            "assignment_weight": self.assignment_weight,
            "total_flow": float(sum(sum(row) for row in self.flow)),
            "min_hub_fixed_cost": float(min(self.hub_fixed_cost)),
            "max_hub_fixed_cost": float(max(self.hub_fixed_cost)),
            "min_hub_capacity": float(min(self.hub_capacity)),
            "max_hub_capacity": float(max(self.hub_capacity)),
        }


def load_instance(path: str | Path) -> HLRPInstance:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return HLRPInstance(data)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and sanity-check a unified HLRP instance JSON.")
    parser.add_argument("--instance", type=str, required=True, help="Path to unified instance JSON")
    parser.add_argument("--hub", type=int, default=0, help="Hub id for the trivial debug solution")
    parser.add_argument("--dump_eval", type=int, default=1, choices=[0, 1], help="Print evaluation details")
    args = parser.parse_args()

    inst = load_instance(args.instance)
    print(json.dumps(inst.summary(), ensure_ascii=False, indent=2))

    sol = inst.build_trivial_solution(hub=args.hub)
    res = inst.evaluate_solution(sol)
    print(f"[trivial_solution] feasible={res.feasible} total_cost={res.total_cost}")
    if args.dump_eval:
        print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
