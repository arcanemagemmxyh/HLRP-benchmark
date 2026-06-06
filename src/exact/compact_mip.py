# -*- coding: utf-8 -*-
"""
Compact MIP baseline for the unified HLRP benchmark.

Improved version:
- records incumbent and bound histories through DOcplex progress listeners
- supports feasibility-oriented CPLEX parameter presets
- supports auto-discovery of the best heuristic JSON as a MIP start
- keeps the original compact formulation unchanged

Recommended usage:
- ap10 / ap20: exact or near-exact baseline
- ap20 / ap25: use a heuristic warm start and feasibility emphasis
- ap40 / ap50: treat as difficult compact-MIP reference only; use longer time *and* a warm start
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple


def load_instance(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def assignment_cost_matrix(data: Mapping) -> List[List[float]]:
    n = int(data["summary"]["n_nodes"])
    dist = data["distance_matrix"]
    tpn = data.get("node_throughput_normalized")
    if not tpn:
        outb = data["outbound_flow"]
        inb = data["inbound_flow"]
        tp = [float(outb[i]) + float(inb[i]) for i in range(n)]
        mean_tp = sum(tp) / max(1, n)
        if mean_tp <= 1e-12:
            mean_tp = 1.0
        tpn = [x / mean_tp for x in tp]
    aw = float(data["parameters"].get("assignment_weight", 1.0))
    a = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for h in range(n):
            if i == h:
                a[i][h] = 0.0
            else:
                a[i][h] = aw * float(tpn[i]) * float(dist[i][h])
    return a


def throughput_vector(data: Mapping) -> List[float]:
    outb = data["outbound_flow"]
    inb = data["inbound_flow"]
    return [float(outb[i]) + float(inb[i]) for i in range(len(outb))]


def parse_heuristic_solution(path: str) -> Tuple[List[int], Dict[int, int], Dict[int, List[List[int]]]]:
    with open(path, "r", encoding="utf-8") as f:
        s = json.load(f)
    hubs = [int(h) for h in s.get("hubs", s.get("open_hubs", []))]
    assign_raw = s.get("assign", s.get("assignment", {}))
    assign = {int(k): int(v) for k, v in assign_raw.items()}

    tours_raw = s.get("tours", s.get("routes", {}))
    tours: Dict[int, List[List[int]]] = {}
    for hk, route_list in tours_raw.items():
        h = int(hk)
        tours[h] = []
        for route in route_list:
            rr = [int(x) for x in route]
            if len(rr) >= 2 and rr[0] == h and rr[-1] == h:
                rr = rr[1:-1]
            tours[h].append(rr)
    return hubs, assign, tours


def extract_routes_from_solution(n: int, y_sol: Dict[Tuple[int, int, int], float], open_hubs: Iterable[int]) -> Dict[int, List[List[int]]]:
    routes: Dict[int, List[List[int]]] = {}
    open_hubs = list(open_hubs)
    tol = 0.5
    for h in open_hubs:
        succ_from_depot = [j for j in range(n) if j != h and y_sol.get((h, h, j), 0.0) > tol]
        succ: Dict[int, int] = {}
        for i in range(n):
            if i == h:
                continue
            outs = [j for j in range(n) if j != i and y_sol.get((h, i, j), 0.0) > tol]
            if outs:
                succ[i] = outs[0]
        hub_routes: List[List[int]] = []
        for first in succ_from_depot:
            cur = first
            route = [h]
            seen = set()
            while True:
                if cur == h:
                    route.append(h)
                    break
                if cur in seen:
                    route.append(cur)
                    break
                seen.add(cur)
                route.append(cur)
                nxt = succ.get(cur, h)
                cur = nxt
            if len(route) >= 3:
                hub_routes.append(route)
        routes[h] = hub_routes
    return routes


def _safe_float(v) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def find_best_heuristic_start(search_dir: str, instance_name: str) -> Optional[str]:
    if not search_dir:
        return None
    root = Path(search_dir)
    if not root.exists():
        return None

    best_path: Optional[Path] = None
    best_obj: Optional[float] = None
    for p in root.rglob("*.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            inst = d.get("instance") or d.get("instance_name") or p.stem
            if str(inst) != str(instance_name):
                continue
            feasible = d.get("final_feasible", d.get("feasible", True))
            if not feasible:
                continue
            obj = d.get("final_obj", d.get("objective"))
            objf = _safe_float(obj)
            if objf is None:
                continue
            if best_obj is None or objf < best_obj - 1e-9:
                best_obj = objf
                best_path = p
        except Exception:
            continue
    return str(best_path) if best_path else None


def build_and_solve(args: argparse.Namespace) -> Dict:
    try:
        from docplex.mp.model import Model
        from docplex.mp.progress import ProgressClock, ProgressListener, SolutionListener
    except Exception as exc:
        raise RuntimeError(
            "docplex is required. Install docplex and ensure the CPLEX Python API is available in this environment."
        ) from exc

    class IncumbentRecorder(SolutionListener):
        def __init__(self):
            super().__init__(clock=ProgressClock.Objective)
            self.improvement_history: List[Dict] = []
            self.incumbent_history: List[Dict] = []
            self.first_obj: Optional[float] = None
            self.best_obj: Optional[float] = None
            self.t_first: Optional[float] = None
            self.t_best: Optional[float] = None
            self.t_last_improvement: Optional[float] = None

        def notify_solution(self, sol):  # type: ignore[override]
            pdata = self.current_progress_data
            t = _safe_float(getattr(pdata, "time", None))
            if t is None:
                t = 0.0
            obj = _safe_float(getattr(sol, "objective_value", None))
            if obj is None:
                return
            self.incumbent_history.append({
                "t_s": t,
                "obj": obj,
                "best_bound": _safe_float(getattr(pdata, "best_bound", None)),
                "gap": _safe_float(getattr(pdata, "mip_gap", None)),
            })
            if self.first_obj is None:
                self.first_obj = obj
                self.t_first = t
            if self.best_obj is None or obj < self.best_obj - 1e-9:
                self.best_obj = obj
                self.t_best = t
                self.t_last_improvement = t
                self.improvement_history.append({
                    "t_s": t,
                    "step": len(self.improvement_history),
                    "obj": obj,
                })

    class BoundRecorder(ProgressListener):
        def __init__(self):
            super().__init__(clock_arg=ProgressClock.BestBound)
            self.bound_history: List[Dict] = []
            self.last_bound: Optional[float] = None

        def notify_progress(self, progress_data):  # type: ignore[override]
            bb = _safe_float(getattr(progress_data, "best_bound", None))
            t = _safe_float(getattr(progress_data, "time", None))
            if bb is None or t is None:
                return
            if self.last_bound is None or abs(bb - self.last_bound) > 1e-9:
                self.last_bound = bb
                self.bound_history.append({
                    "t_s": t,
                    "best_bound": bb,
                    "gap": _safe_float(getattr(progress_data, "mip_gap", None)),
                    "has_incumbent": bool(getattr(progress_data, "has_incumbent", False)),
                })

    data = load_instance(args.instance_json)
    n = int(data["summary"]["n_nodes"])
    alpha = float(data["parameters"]["alpha"])
    q = int(data["parameters"]["q"])
    dist = [[float(x) for x in row] for row in data["distance_matrix"]]
    hub_fixed = [float(x) for x in data["hub_fixed_cost"]]
    hub_cap = [float(x) for x in data["hub_capacity"]]
    tp = throughput_vector(data)
    assign_cost = assignment_cost_matrix(data)
    flow_edges = [(int(e["from"]), int(e["to"]), float(e["flow"])) for e in data.get("flow_edges", []) if float(e["flow"]) > 1e-12]

    V = list(range(n))
    mdl = Model(name=f"unified_hlrp_{data['instance_name']}")
    mdl.parameters.timelimit = float(args.time_limit)
    mdl.parameters.mip.tolerances.mipgap = float(args.mip_gap)
    if int(args.threads) > 0:
        mdl.parameters.threads = int(args.threads)

    if args.mip_emphasis is not None:
        mdl.parameters.emphasis.mip = int(args.mip_emphasis)
    if args.heuristic_freq is not None:
        mdl.parameters.mip.strategy.heuristicfreq = int(args.heuristic_freq)
    if args.rins_heur is not None:
        mdl.parameters.mip.strategy.rinsheur = int(args.rins_heur)
    if args.fp_heur is not None:
        mdl.parameters.mip.strategy.fpheur = int(args.fp_heur)
    if args.node_file is not None:
        mdl.parameters.mip.strategy.file = int(args.node_file)
    if args.workmem is not None and float(args.workmem) > 0:
        mdl.parameters.workmem = float(args.workmem)

    z = mdl.binary_var_dict(V, name="z")
    x = mdl.binary_var_matrix(V, V, name="x")

    strengthen_mip = int(getattr(args, "strengthen_mip", 0) or 0)
    sparse_equivalent_mip = int(getattr(args, "sparse_equivalent_mip", 0) or 0)
    capacity_impossible = {
        (i, h) for i in V for h in V if i != h and tp[i] > hub_cap[h] + 1e-9
    }

    def assign_pair_possible(i: int, h: int) -> bool:
        return i == h or (i, h) not in capacity_impossible

    def arc_possible(h: int, i: int, j: int) -> bool:
        if i == j:
            return False
        if i != h and not assign_pair_possible(i, h):
            return False
        if j != h and not assign_pair_possible(j, h):
            return False
        return True

    arc_keys: List[Tuple[int, int, int]] = []
    for h in V:
        for i in V:
            for j in V:
                if sparse_equivalent_mip and not arc_possible(h, i, j):
                    continue
                if not sparse_equivalent_mip and i == j:
                    continue
                arc_keys.append((h, i, j))
    y = mdl.binary_var_dict(arc_keys, name="y")

    u_keys = [
        (h, i)
        for h in V
        for i in V
        if i != h and (not sparse_equivalent_mip or assign_pair_possible(i, h))
    ]
    u = mdl.continuous_var_dict(u_keys, lb=0.0, ub=float(q), name="u")

    hk_pairs = [(h, k) for h in V for k in V if h != k and dist[h][k] > 1e-12]
    w_keys = [
        (i, j, h, k)
        for (i, j, fij) in flow_edges
        for (h, k) in hk_pairs
        if not sparse_equivalent_mip
        or (i != j and assign_pair_possible(i, h) and assign_pair_possible(j, k))
    ]
    w = mdl.continuous_var_dict(w_keys, lb=0.0, ub=1.0, name="w")
    strengthening_counts = {
        "aggregate_capacity": 0,
        "infeasible_assignment": 0,
        "route_count": 0,
        "interhub_w_aggregation": 0,
    }
    sparse_counts = {
        "capacity_impossible_assignments": len(capacity_impossible),
        "arc_keys_removed": n * n * (n - 1) - len(arc_keys),
        "u_keys_removed": n * (n - 1) - len(u_keys),
        "w_keys_removed": len(flow_edges) * len(hk_pairs) - len(w_keys),
    }

    mdl.add_constraint(mdl.sum(z[h] for h in V) >= 1, ctname="at_least_one_hub")
    for i in V:
        mdl.add_constraint(mdl.sum(x[i, h] for h in V) == 1, ctname=f"assign_once_{i}")
        mdl.add_constraint(x[i, i] == z[i], ctname=f"self_assign_if_open_{i}")
        for h in V:
            mdl.add_constraint(x[i, h] <= z[h], ctname=f"assign_only_to_open_{i}_{h}")
            if h != i:
                mdl.add_constraint(x[i, h] <= 1 - z[i], ctname=f"open_node_not_customer_{i}_{h}")

    for h in V:
        mdl.add_constraint(
            mdl.sum(tp[i] * x[i, h] for i in V if i != h) <= hub_cap[h] * z[h],
            ctname=f"hub_capacity_{h}",
        )

    if strengthen_mip:
        mdl.add_constraint(
            mdl.sum(tp[i] * (1 - z[i]) for i in V) <= mdl.sum(hub_cap[h] * z[h] for h in V),
            ctname="aggregate_hub_capacity",
        )
        strengthening_counts["aggregate_capacity"] += 1
        for i, h in sorted(capacity_impossible):
            mdl.add_constraint(x[i, h] == 0, ctname=f"infeasible_assign_capacity_{i}_{h}")
            strengthening_counts["infeasible_assignment"] += 1

    for h in V:
        for i in V:
            if i == h:
                continue
            mdl.add_constraint(
                mdl.sum(y[h, i, j] for j in V if (h, i, j) in y) == x[i, h],
                ctname=f"route_out_{h}_{i}",
            )
            mdl.add_constraint(
                mdl.sum(y[h, j, i] for j in V if (h, j, i) in y) == x[i, h],
                ctname=f"route_in_{h}_{i}",
            )
        mdl.add_constraint(
            mdl.sum(y[h, h, j] for j in V if (h, h, j) in y)
            == mdl.sum(y[h, i, h] for i in V if (h, i, h) in y),
            ctname=f"depot_balance_{h}",
        )
        if strengthen_mip:
            assigned_count = mdl.sum(x[i, h] for i in V if i != h)
            route_count = mdl.sum(y[h, h, j] for j in V if (h, h, j) in y)
            mdl.add_constraint(route_count >= assigned_count / float(q), ctname=f"route_count_lb_card_{h}")
            strengthening_counts["route_count"] += 1
            for i in V:
                if i != h:
                    mdl.add_constraint(route_count >= x[i, h], ctname=f"route_count_lb_any_{h}_{i}")
                    strengthening_counts["route_count"] += 1

    for (h, i, j) in arc_keys:
        if i != h:
            mdl.add_constraint(y[h, i, j] <= x[i, h], ctname=f"link_from_{h}_{i}_{j}")
        if j != h:
            mdl.add_constraint(y[h, i, j] <= x[j, h], ctname=f"link_to_{h}_{i}_{j}")
        mdl.add_constraint(y[h, i, j] <= z[h], ctname=f"link_hub_open_{h}_{i}_{j}")

    for (h, i) in u_keys:
        mdl.add_constraint(u[h, i] >= x[i, h], ctname=f"u_lb_{h}_{i}")
        mdl.add_constraint(u[h, i] <= q * x[i, h], ctname=f"u_ub_{h}_{i}")

    for h in V:
        cust = [i for i in V if i != h]
        for i in cust:
            for j in cust:
                if (h, i, j) not in y:
                    continue
                mdl.add_constraint(
                    u[h, i] - u[h, j] + q * y[h, i, j] <= q - 1 + q * (2 - x[i, h] - x[j, h]),
                    ctname=f"mtz_{h}_{i}_{j}",
                )

    for (i, j, fij) in flow_edges:
        for (h, k) in hk_pairs:
            if (i, j, h, k) not in w:
                continue
            var = w[i, j, h, k]
            mdl.add_constraint(var <= x[i, h], ctname=f"w_ub1_{i}_{j}_{h}_{k}")
            mdl.add_constraint(var <= x[j, k], ctname=f"w_ub2_{i}_{j}_{h}_{k}")
            mdl.add_constraint(var >= x[i, h] + x[j, k] - 1, ctname=f"w_lb_{i}_{j}_{h}_{k}")

    if strengthen_mip:
        hk_set = set(hk_pairs)
        full_offdiag_pairs = len(hk_pairs) == n * (n - 1)
        for (i, j, fij) in flow_edges:
            for h in V:
                out_terms = [w[i, j, h, k] for k in V if (h, k) in hk_set and (i, j, h, k) in w]
                if out_terms:
                    out_sum = mdl.sum(out_terms)
                    mdl.add_constraint(out_sum <= x[i, h], ctname=f"w_out_agg_ub_{i}_{j}_{h}")
                    strengthening_counts["interhub_w_aggregation"] += 1
                    if full_offdiag_pairs:
                        mdl.add_constraint(out_sum >= x[i, h] - x[j, h], ctname=f"w_out_agg_lb_{i}_{j}_{h}")
                        strengthening_counts["interhub_w_aggregation"] += 1
            for k in V:
                in_terms = [w[i, j, h, k] for h in V if (h, k) in hk_set and (i, j, h, k) in w]
                if in_terms:
                    in_sum = mdl.sum(in_terms)
                    mdl.add_constraint(in_sum <= x[j, k], ctname=f"w_in_agg_ub_{i}_{j}_{k}")
                    strengthening_counts["interhub_w_aggregation"] += 1
                    if full_offdiag_pairs:
                        mdl.add_constraint(in_sum >= x[j, k] - x[i, k], ctname=f"w_in_agg_lb_{i}_{j}_{k}")
                        strengthening_counts["interhub_w_aggregation"] += 1

    hub_term = mdl.sum(hub_fixed[h] * z[h] for h in V)
    assign_term = mdl.sum(assign_cost[i][h] * x[i, h] for i in V for h in V if i != h and assign_cost[i][h] != 0.0)
    route_term = mdl.sum(dist[i][j] * y[h, i, j] for (h, i, j) in arc_keys if dist[i][j] != 0.0)
    interhub_term = mdl.sum(alpha * fij * dist[h][k] * w[i, j, h, k] for (i, j, fij) in flow_edges for (h, k) in hk_pairs if (i, j, h, k) in w)
    mdl.minimize(hub_term + assign_term + route_term + interhub_term)

    mip_start_path = args.mip_start_json or ""
    if not mip_start_path and args.auto_mip_start_dir:
        auto_path = find_best_heuristic_start(args.auto_mip_start_dir, data["instance_name"])
        if auto_path:
            mip_start_path = auto_path
            print(f"[info] auto-selected mip start: {mip_start_path}")

    if mip_start_path:
        try:
            hs_hubs, hs_assign, hs_tours = parse_heuristic_solution(mip_start_path)
            ms = mdl.new_solution()
            hub_set = set(hs_hubs)
            for h in V:
                ms.add_var_value(z[h], 1 if h in hub_set else 0)
            full_assign = {i: (i if i in hub_set else hs_assign.get(i, None)) for i in V}
            for i in V:
                hi = full_assign.get(i, None)
                if hi is None:
                    continue
                for h in V:
                    ms.add_var_value(x[i, h], 1 if h == hi else 0)
            for h, route_list in hs_tours.items():
                for route in route_list:
                    rr = [int(xv) for xv in route]
                    if len(rr) >= 2 and rr[0] == h and rr[-1] == h:
                        rr = rr[1:-1]
                    if not rr:
                        continue
                    prev = h
                    pos = 0
                    for c in rr:
                        if (h, prev, c) in y:
                            ms.add_var_value(y[h, prev, c], 1)
                        pos += 1
                        if (h, c) in u:
                            ms.add_var_value(u[h, c], min(q, pos))
                        prev = c
                    if (h, prev, h) in y:
                        ms.add_var_value(y[h, prev, h], 1)
            mdl.add_mip_start(ms, write_level="nonzero_discrete_vars")
        except Exception as exc:
            print(f"[warn] failed to add mip start from {mip_start_path}: {exc}")

    inc_rec = IncumbentRecorder()
    bnd_rec = BoundRecorder()
    mdl.add_progress_listener(inc_rec)
    mdl.add_progress_listener(bnd_rec)

    t0 = time.time()
    sol = mdl.solve(log_output=bool(args.log_output))
    wall = time.time() - t0

    details = mdl.solve_details
    status = str(mdl.solve_status)
    best_bound = _safe_float(getattr(details, "best_bound", None))
    final_gap = _safe_float(getattr(details, "mip_relative_gap", None))
    stopped_by_time = bool(float(args.time_limit) > 0 and wall >= float(args.time_limit) - 1e-6)

    result: Dict = {
        "instance": data["instance_name"],
        "source": str(args.instance_json),
        "status": status,
        "time_limit": float(args.time_limit),
        "mip_gap_target": float(args.mip_gap),
        "solve_wall_s": wall,
        "best_bound": best_bound,
        "mip_relative_gap": final_gap,
        "n_nodes": n,
        "q": q,
        "strengthen_mip": strengthen_mip,
        "sparse_equivalent_mip": sparse_equivalent_mip,
        "strengthening_counts": strengthening_counts,
        "sparse_counts": sparse_counts,
        "feasible": sol is not None,
        "perf": {
            "algorithm": "MIP",
            "seed": None,
            "instance": data["instance_name"],
            "step_unit": "mip_progress",
            "stopped_by_time": stopped_by_time,
            "time_limit_s": float(args.time_limit),
            "total_wall_s": wall,
            "runtime_utilization": wall / max(float(args.time_limit), 1e-9),
            "first_feasible_obj": inc_rec.first_obj,
            "best_feasible_obj": inc_rec.best_obj,
            "time_to_first_feasible_s": inc_rec.t_first,
            "time_to_best_feasible_s": inc_rec.t_best,
            "time_to_last_improvement_s": inc_rec.t_last_improvement,
            "first_feasible_step": 0 if inc_rec.first_obj is not None else None,
            "best_feasible_step": None if inc_rec.best_obj is None else max(0, len(inc_rec.improvement_history) - 1),
            "last_improvement_step": None if inc_rec.best_obj is None else max(0, len(inc_rec.improvement_history) - 1),
            "num_improvements": len(inc_rec.improvement_history),
            "final_feasible": bool(sol is not None),
        },
        "improvement_history": inc_rec.improvement_history[:],
        "mip_perf": {
            "first_incumbent_obj": inc_rec.first_obj,
            "best_incumbent_obj": inc_rec.best_obj,
            "best_bound": best_bound,
            "final_gap": final_gap,
            "proven_optimal": bool(final_gap is not None and final_gap <= 1e-12),
            "time_to_first_incumbent_s": inc_rec.t_first,
            "time_to_best_incumbent_s": inc_rec.t_best,
            "time_to_best_bound_s": bnd_rec.bound_history[-1]["t_s"] if bnd_rec.bound_history else None,
            "time_to_optimality_s": wall if (final_gap is not None and final_gap <= 1e-12) else None,
        },
        "incumbent_history": inc_rec.incumbent_history[:],
        "best_bound_history": bnd_rec.bound_history[:],
        "selected_mip_start_json": mip_start_path if mip_start_path else None,
        "cplex_params": {
            "mip_emphasis": int(args.mip_emphasis) if args.mip_emphasis is not None else None,
            "heuristic_freq": int(args.heuristic_freq) if args.heuristic_freq is not None else None,
            "rins_heur": int(args.rins_heur) if args.rins_heur is not None else None,
            "fp_heur": int(args.fp_heur) if args.fp_heur is not None else None,
            "node_file": int(args.node_file) if args.node_file is not None else None,
            "workmem": float(args.workmem) if args.workmem is not None else None,
            "strengthen_mip": strengthen_mip,
            "sparse_equivalent_mip": sparse_equivalent_mip,
        },
    }

    if sol is None:
        return result

    obj = float(sol.objective_value)
    if not result["improvement_history"]:
        result["improvement_history"] = [{"t_s": wall, "step": 0, "obj": obj}]
        result["incumbent_history"] = [{"t_s": wall, "obj": obj, "best_bound": best_bound, "gap": final_gap}]
        result["perf"]["first_feasible_obj"] = obj
        result["perf"]["best_feasible_obj"] = obj
        result["perf"]["time_to_first_feasible_s"] = wall
        result["perf"]["time_to_best_feasible_s"] = wall
        result["perf"]["time_to_last_improvement_s"] = wall
        result["perf"]["first_feasible_step"] = 0
        result["perf"]["best_feasible_step"] = 0
        result["perf"]["last_improvement_step"] = 0
        result["perf"]["num_improvements"] = 1
        result["mip_perf"]["first_incumbent_obj"] = obj
        result["mip_perf"]["best_incumbent_obj"] = obj
        result["mip_perf"]["time_to_first_incumbent_s"] = wall
        result["mip_perf"]["time_to_best_incumbent_s"] = wall

    z_sol = {h: z[h].solution_value for h in V}
    x_sol = {(i, h): x[i, h].solution_value for i in V for h in V}
    y_sol = {(h, i, j): y[h, i, j].solution_value for (h, i, j) in arc_keys}
    w_sol = {(i, j, h, k): w[i, j, h, k].solution_value for (i, j, h, k) in w_keys}

    open_hubs = [h for h in V if z_sol[h] > 0.5]
    assign = {}
    for i in V:
        best_h = max(V, key=lambda h: x_sol[i, h])
        if i not in open_hubs:
            assign[i] = int(best_h)
    tours = extract_routes_from_solution(n, y_sol, open_hubs)

    hub_cost = sum(hub_fixed[h] for h in open_hubs)
    assignment_cost_val = sum(assign_cost[i][h] * x_sol[i, h] for i in V for h in V if i != h)
    route_cost_val = sum(dist[i][j] * y_sol[h, i, j] for (h, i, j) in arc_keys)
    interhub_cost_val = sum(
        alpha * fij * dist[h][k] * w_sol.get((i, j, h, k), 0.0)
        for (i, j, fij) in flow_edges
        for (h, k) in hk_pairs
    )

    result.update(
        {
            "objective": obj,
            "hubs": open_hubs,
            "assign": {str(k): int(v) for k, v in assign.items()},
            "tours": {str(h): routes for h, routes in tours.items()},
            "costs": {
                "route": route_cost_val,
                "assignment": assignment_cost_val,
                "interhub": interhub_cost_val,
                "vehicle": 0.0,
                "hub": hub_cost,
            },
            "mip_stats": {
                "status": status,
                "objective": obj,
                "best_bound": best_bound,
                "mip_relative_gap": final_gap,
                "solve_time_s": wall,
                "problem_num_vars": mdl.number_of_variables,
                "problem_num_constraints": mdl.number_of_constraints,
            },
        }
    )
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Compact CPLEX MIP for the unified HLRP benchmark")
    p.add_argument("--instance_json", type=str, required=True)
    p.add_argument("--out_json", type=str, required=True)
    p.add_argument("--time_limit", type=float, default=600.0)
    p.add_argument("--mip_gap", type=float, default=0.01)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--mip_start_json", type=str, default="")
    p.add_argument("--auto_mip_start_dir", type=str, default="")
    p.add_argument("--log_output", type=int, default=1)
    p.add_argument("--mip_emphasis", type=int, default=1, help="0 balanced, 1 feasibility, 2 optimality, 3 best bound, 4 hidden feasibility")
    p.add_argument("--heuristic_freq", type=int, default=10)
    p.add_argument("--rins_heur", type=int, default=10)
    p.add_argument("--fp_heur", type=int, default=1)
    p.add_argument("--node_file", type=int, default=3, help="0 no node file, 1/2/3 increasingly disk-oriented")
    p.add_argument("--workmem", type=float, default=4096.0)
    p.add_argument("--strengthen_mip", type=int, default=0, help="Add safe valid inequalities for route count and inter-hub flow linearization")
    p.add_argument("--sparse_equivalent_mip", type=int, default=0, choices=[0, 1], help="Omit variables that are fixed to zero by capacity feasibility without changing the model semantics")
    args = p.parse_args()

    res = build_and_solve(args)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({
        "instance": res.get("instance"),
        "status": res.get("status"),
        "objective": res.get("objective"),
        "best_bound": res.get("best_bound"),
        "gap": res.get("mip_relative_gap"),
        "mip_start": res.get("selected_mip_start_json"),
        "out_json": str(out_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
