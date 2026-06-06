# -*- coding: utf-8 -*-
r"""
chclrp_ts_v2.py

Unified tabu search for the common HLRP benchmark v2.

Main differences from the earlier integration version
- still keeps the permutation-based TS framework
- supports unified benchmark JSON and original csv + param_json input
- removes Karimi-specific Gamma and T_timebound constraints
- adds mild assignment cost
- allows opening additional hubs proactively instead of only when capacity is violated
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _now_s() -> float:
    return time.perf_counter()


def _time_exceeded(deadline_s: Optional[float]) -> bool:
    return deadline_s is not None and _now_s() >= deadline_s - 1e-9


def load_matrix_from_edges(path: Path, value_col: str) -> np.ndarray:
    df = pd.read_csv(path)
    if not {"fromnode", "tonode", value_col}.issubset(df.columns):
        raise ValueError(f"{path} must have columns: fromnode, tonode, {value_col}")
    n = int(max(df["fromnode"].max(), df["tonode"].max()) + 1)
    mat = np.zeros((n, n), dtype=float)
    for _, r in df.iterrows():
        i = int(r["fromnode"])
        j = int(r["tonode"])
        mat[i, j] = float(r[value_col])
    return mat


def _pick_col(df: pd.DataFrame, cols: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for c in cols:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def load_unified_json_instance(instance_json: str, q_override: Optional[int] = None) -> dict:
    with open(instance_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    n = int(data["summary"]["n_nodes"])
    C = np.array(data["distance_matrix"], dtype=float)
    W = np.array(data["flow_matrix"], dtype=float)
    if C.shape != (n, n) or W.shape != (n, n):
        raise ValueError("distance_matrix or flow_matrix shape mismatch")

    O = np.array(data["outbound_flow"], dtype=float)
    D = np.array(data["inbound_flow"], dtype=float)
    TP = np.array(data.get("node_throughput", O + D), dtype=float)
    TP_norm = np.array(data.get("node_throughput_normalized", []), dtype=float)
    if TP_norm.size == 0:
        mean_tp = float(np.mean(TP)) if n > 0 else 1.0
        if mean_tp <= 1e-12:
            mean_tp = 1.0
        TP_norm = TP / mean_tp

    q = int(data["parameters"]["q"] if q_override is None else q_override)
    return {
        "mode": "unified_json",
        "name": str(data.get("instance_name", Path(instance_json).stem)),
        "n": n,
        "N": list(range(n)),
        "C": C,
        "W": W,
        "O": O,
        "D": D,
        "TP": TP,
        "TP_norm": TP_norm,
        "Fhub": np.array(data["hub_fixed_cost"], dtype=float),
        "Lambda": np.array(data["hub_capacity"], dtype=float),
        "alpha": float(data["parameters"].get("alpha", 0.75)),
        "q": q,
        "F_vehicle": float(data["parameters"].get("vehicle_fixed_cost", 0.0)),
        "assignment_weight": float(data["parameters"].get("assignment_weight", 0.0)),
        "source": str(instance_json),
    }


def load_csv_param_instance(data_dir: str, prefix: str, param_json: str, cap_type: str, cost_type: str, alpha: float, q: int, vehicle_fixed_cost: Optional[float], assignment_weight: float) -> dict:
    d = Path(data_dir)
    nodes_path = d / f"{prefix}_nodes.csv"
    c_path = d / f"{prefix}_c.csv"
    w_path = d / f"{prefix}_w.csv"
    if not nodes_path.exists() or not c_path.exists() or not w_path.exists():
        raise FileNotFoundError("missing csv inputs")
    if not param_json:
        raise ValueError("csv mode requires --param_json")

    nodes = pd.read_csv(nodes_path).sort_values("ID").reset_index(drop=True)
    C = load_matrix_from_edges(c_path, "c")
    W = load_matrix_from_edges(w_path, "w")
    np.fill_diagonal(W, 0.0)
    n = C.shape[0]

    with open(param_json, "r", encoding="utf-8") as f:
        pj = json.load(f)

    cap_type = cap_type.upper()
    cost_type = cost_type.upper()
    keyL = "Lambda_L" if cap_type == "L" else "Lambda_T"
    keyF = "Fhub_L" if cost_type == "L" else "Fhub_T"
    Lambda = np.array(pj.get(keyL, pj.get("Lambda", [])), dtype=float).reshape(-1)
    Fhub = np.array(pj.get(keyF, pj.get("Fhub", [])), dtype=float).reshape(-1)

    if Lambda.size == 0:
        cap_col = _pick_col(nodes, ["cap_L" if cap_type == "L" else "cap_T", "Lambda_L", "Lambda_T"])
        if cap_col is not None:
            Lambda = nodes[cap_col].to_numpy(dtype=float)
    if Fhub.size == 0:
        f_col = _pick_col(nodes, ["F_L" if cost_type == "L" else "F_T", "Fhub_L", "Fhub_T"])
        if f_col is not None:
            Fhub = nodes[f_col].to_numpy(dtype=float)
    if len(Lambda) != n or len(Fhub) != n:
        raise ValueError("Fhub/Lambda length mismatch with node count")

    O = W.sum(axis=1)
    D = W.sum(axis=0)
    TP = O + D
    mean_tp = float(np.mean(TP)) if n > 0 else 1.0
    if mean_tp <= 1e-12:
        mean_tp = 1.0
    TP_norm = TP / mean_tp

    return {
        "mode": "csv_param",
        "name": prefix,
        "n": n,
        "N": list(range(n)),
        "C": C,
        "W": W,
        "O": O,
        "D": D,
        "TP": TP,
        "TP_norm": TP_norm,
        "Fhub": np.array(Fhub, dtype=float),
        "Lambda": np.array(Lambda, dtype=float),
        "alpha": float(alpha),
        "q": int(q),
        "F_vehicle": float(0.0 if vehicle_fixed_cost is None else vehicle_fixed_cost),
        "assignment_weight": float(assignment_weight),
        "source": str(d),
    }


def cycle_cost(C: np.ndarray, tour: List[int]) -> float:
    return float(sum(C[tour[i], tour[i + 1]] for i in range(len(tour) - 1)))


def route_cost(C: np.ndarray, hub: int, seq: List[int]) -> float:
    if not seq:
        return 0.0
    return cycle_cost(C, [hub] + seq + [hub])


def assignment_cost_single(inst: dict, customer: int, hub: int) -> float:
    if customer == hub:
        return 0.0
    return float(inst["assignment_weight"] * inst["TP_norm"][customer] * inst["C"][customer, hub])


def best_insertion_position(C: np.ndarray, hub: int, seq: List[int], node: int) -> Tuple[float, List[int]]:
    best_delta = float("inf")
    best_seq = None
    old_cost = route_cost(C, hub, seq)
    for pos in range(len(seq) + 1):
        new_seq = seq[:pos] + [node] + seq[pos:]
        delta = route_cost(C, hub, new_seq) - old_cost
        if delta < best_delta:
            best_delta = delta
            best_seq = new_seq
    return best_delta, best_seq


def check_hub_cap_after_add(inst: dict, hub: int, assigned: List[int]) -> bool:
    load = float(sum(inst["O"][i] + inst["D"][i] for i in assigned))
    return load <= float(inst["Lambda"][hub]) + 1e-9


def compute_interhub_cost(inst: dict, assign: Dict[int, int]) -> float:
    C = inst["C"]
    W = inst["W"]
    alpha = float(inst["alpha"])
    n = inst["n"]
    cost = 0.0
    for i in range(n):
        hi = int(assign.get(i, i))
        for j in range(n):
            w = float(W[i, j])
            if w == 0.0:
                continue
            hj = int(assign.get(j, j))
            cost += alpha * w * C[hi, hj]
    return float(cost)


def incremental_interhub_cost_for_new_node(inst: dict, v: int, hub: int, assign_so_far: Dict[int, int]) -> float:
    C = inst["C"]
    W = inst["W"]
    alpha = float(inst["alpha"])
    delta = 0.0
    for j, hj in assign_so_far.items():
        hj = int(hj)
        w1 = float(W[v, j])
        if w1 != 0.0:
            delta += alpha * w1 * C[hub, hj]
        w2 = float(W[j, v])
        if w2 != 0.0:
            delta += alpha * w2 * C[hj, hub]
    return float(delta)


@dataclass
class HubSolution:
    hubs: List[int]
    assign: Dict[int, int]
    tours: Dict[int, List[List[int]]]
    route_cost: float
    assignment_cost: float
    interhub_cost: float
    vehicle_cost: float
    hub_cost: float
    feasible: bool
    violations: List[str]

    @property
    def objective(self) -> float:
        return self.route_cost + self.assignment_cost + self.interhub_cost + self.vehicle_cost + self.hub_cost


def evaluate_decoded_solution(inst: dict, hubs: List[int], assign: Dict[int, int], tours_cust: Dict[int, List[List[int]]]) -> HubSolution:
    C = inst["C"]
    n = inst["n"]
    q = int(inst["q"])
    F_vehicle = float(inst["F_vehicle"])
    open_hubs = sorted(set(int(h) for h in hubs))
    hub_set = set(open_hubs)
    violations: List[str] = []
    assign2 = {int(i): int(h) for i, h in assign.items()}
    for h in open_hubs:
        assign2[h] = h
    customers = [i for i in range(n) if i not in hub_set]
    for c in customers:
        if c not in assign2:
            violations.append(f"customer {c} has no assignment")
        elif assign2[c] not in hub_set:
            violations.append(f"customer {c} assigned to closed hub {assign2[c]}")

    route_cost_total = 0.0
    assignment_cost_total = 0.0
    vehicle_count = 0
    seen = set()
    full_tours = {h: [] for h in open_hubs}
    assigned_to_hub = {h: [] for h in open_hubs}

    for c in customers:
        h = assign2.get(c)
        if h in assigned_to_hub:
            assigned_to_hub[h].append(c)
            assignment_cost_total += assignment_cost_single(inst, c, h)

    for h in open_hubs:
        for seq in tours_cust.get(h, []):
            if not seq:
                violations.append(f"empty route under hub {h}")
                continue
            if len(seq) > q:
                violations.append(f"route under hub {h} exceeds q={q}: len={len(seq)}")
            local_seen = set()
            for c in seq:
                if c in hub_set:
                    violations.append(f"open hub {c} appears in a route under hub {h}")
                if c in local_seen:
                    violations.append(f"customer {c} repeated within one route under hub {h}")
                if assign2.get(c) != h:
                    violations.append(f"customer {c} routed by hub {h} but assigned to {assign2.get(c)}")
                if c in seen:
                    violations.append(f"customer {c} appears in more than one route")
                local_seen.add(c)
                seen.add(c)
            tour = [h] + list(seq) + [h]
            full_tours[h].append(tour)
            route_cost_total += cycle_cost(C, tour)
            vehicle_count += 1

    missing = sorted(set(customers) - seen)
    if missing:
        violations.append(f"customers missing from routes: {missing}")

    for h in open_hubs:
        load = float(sum(inst["O"][i] + inst["D"][i] for i in assigned_to_hub[h]))
        if load > float(inst["Lambda"][h]) + 1e-9:
            violations.append(f"hub {h} capacity violated: load={load:.6f} > cap={inst['Lambda'][h]:.6f}")

    hub_cost = float(sum(inst["Fhub"][h] for h in open_hubs))
    vehicle_cost = float(vehicle_count * F_vehicle)
    interhub_cost = compute_interhub_cost(inst, assign2)
    feasible = len(violations) == 0
    return HubSolution(open_hubs, assign2, full_tours, float(route_cost_total), float(assignment_cost_total), float(interhub_cost), float(vehicle_cost), float(hub_cost), feasible, violations)


def two_opt_seq(C: np.ndarray, hub: int, seq: List[int], deadline_s: Optional[float] = None) -> List[int]:
    if len(seq) < 4:
        return seq[:]
    best = seq[:]
    improved = True
    while improved:
        if _time_exceeded(deadline_s):
            break
        improved = False
        base_cost = route_cost(C, hub, best)
        for i in range(len(best) - 1):
            if _time_exceeded(deadline_s):
                return best
            for j in range(i + 1, len(best)):
                if _time_exceeded(deadline_s):
                    return best
                cand = best[:i] + list(reversed(best[i:j + 1])) + best[j + 1:]
                cc = route_cost(C, hub, cand)
                if cc < base_cost - 1e-9:
                    best = cand
                    improved = True
                    break
            if improved:
                break
    return best


def local_search_within_hub(inst: dict, hub: int, tour_seqs: List[List[int]], deadline_s: Optional[float] = None) -> List[List[int]]:
    C = inst["C"]
    q = int(inst["q"])
    Fv = float(inst["F_vehicle"])
    tours = [seq[:] for seq in tour_seqs if seq]
    tours = [two_opt_seq(C, hub, seq, deadline_s=deadline_s) for seq in tours]
    improved = True
    while improved:
        if _time_exceeded(deadline_s):
            break
        improved = False
        for a_idx in range(len(tours)):
            if _time_exceeded(deadline_s):
                return [two_opt_seq(C, hub, seq, deadline_s=deadline_s) for seq in tours]
            for pos in range(len(tours[a_idx])):
                if _time_exceeded(deadline_s):
                    return [two_opt_seq(C, hub, seq, deadline_s=deadline_s) for seq in tours]
                node = tours[a_idx][pos]
                src_removed = tours[a_idx][:pos] + tours[a_idx][pos + 1:]
                old_total = sum(route_cost(C, hub, seq) for seq in tours) + len(tours) * Fv
                for b_idx in range(len(tours)):
                    if _time_exceeded(deadline_s):
                        return [two_opt_seq(C, hub, seq, deadline_s=deadline_s) for seq in tours]
                    base_b = src_removed if b_idx == a_idx else tours[b_idx]
                    if len(base_b) >= q:
                        continue
                    for ins in range(len(base_b) + 1):
                        if _time_exceeded(deadline_s):
                            return [two_opt_seq(C, hub, seq, deadline_s=deadline_s) for seq in tours]
                        cand_b = base_b[:ins] + [node] + base_b[ins:]
                        new_tours = []
                        for idx, seq in enumerate(tours):
                            if idx == a_idx and idx == b_idx:
                                if cand_b:
                                    new_tours.append(cand_b)
                            elif idx == a_idx:
                                if src_removed:
                                    new_tours.append(src_removed)
                            elif idx == b_idx:
                                if cand_b:
                                    new_tours.append(cand_b)
                            else:
                                new_tours.append(seq)
                        new_total = sum(route_cost(C, hub, seq) for seq in new_tours) + len(new_tours) * Fv
                        if new_total < old_total - 1e-9:
                            tours = [seq for seq in new_tours if seq]
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        for i in range(len(tours)):
            if _time_exceeded(deadline_s):
                return [two_opt_seq(C, hub, seq, deadline_s=deadline_s) for seq in tours]
            for j in range(i + 1, len(tours)):
                if _time_exceeded(deadline_s):
                    return [two_opt_seq(C, hub, seq, deadline_s=deadline_s) for seq in tours]
                a = tours[i]
                b = tours[j]
                if len(a) + len(b) > q:
                    continue
                cand = a[:]
                for node in b:
                    if _time_exceeded(deadline_s):
                        return [two_opt_seq(C, hub, seq, deadline_s=deadline_s) for seq in tours]
                    _, cand = best_insertion_position(C, hub, cand, node)
                old_val = route_cost(C, hub, a) + route_cost(C, hub, b) + 2 * Fv
                new_val = route_cost(C, hub, cand) + 1 * Fv
                if new_val < old_val - 1e-9:
                    tours = [seq for idx, seq in enumerate(tours) if idx not in (i, j)] + [cand]
                    improved = True
                    break
            if improved:
                break
    return [two_opt_seq(C, hub, seq, deadline_s=deadline_s) for seq in tours]


def decode_enhanced_feasible(inst: dict, perm: List[int], deadline_s: Optional[float] = None) -> Optional[HubSolution]:
    first_hub = int(perm[0])
    hubs = [first_hub]
    tours_cust = {first_hub: []}
    assign = {first_hub: first_hub}
    assigned_to_hub = {first_hub: []}

    for v_raw in perm[1:]:
        if _time_exceeded(deadline_s):
            return None
        v = int(v_raw)
        best_choice = None
        for h in hubs:
            cand_assigned = assigned_to_hub[h] + [v]
            if not check_hub_cap_after_add(inst, h, cand_assigned):
                continue
            best_route_delta = None
            best_route_seq_idx = None
            best_new_seq = None
            for idx, seq in enumerate(tours_cust[h]):
                if len(seq) >= int(inst["q"]):
                    continue
                delta, new_seq = best_insertion_position(inst["C"], h, seq, v)
                if best_route_delta is None or delta < best_route_delta - 1e-9:
                    best_route_delta = delta
                    best_route_seq_idx = idx
                    best_new_seq = new_seq
            if best_route_delta is None:
                best_route_delta = route_cost(inst["C"], h, [v]) + float(inst["F_vehicle"])
                best_route_seq_idx = None
                best_new_seq = [v]
            cand_val = float(best_route_delta) + assignment_cost_single(inst, v, h) + incremental_interhub_cost_for_new_node(inst, v, h, assign)
            cand = ("assign_existing", cand_val, h, best_route_seq_idx, best_new_seq)
            if best_choice is None or cand_val < best_choice[1] - 1e-9:
                best_choice = cand
        open_val = float(inst["Fhub"][v]) + incremental_interhub_cost_for_new_node(inst, v, v, assign)
        cand = ("open_new_hub", open_val, v, None, None)
        if best_choice is None or open_val < best_choice[1] - 1e-9:
            best_choice = cand
        if best_choice is None:
            return None
        kind, _, h, seq_idx, new_seq = best_choice
        if kind == "open_new_hub":
            if v not in tours_cust:
                hubs.append(v)
                tours_cust[v] = []
                assigned_to_hub[v] = []
            assign[v] = v
        else:
            assign[v] = h
            assigned_to_hub[h].append(v)
            if seq_idx is None:
                tours_cust[h].append([v])
            else:
                tours_cust[h][seq_idx] = new_seq

    for h in list(tours_cust.keys()):
        tours_cust[h] = local_search_within_hub(inst, h, tours_cust[h], deadline_s=deadline_s)
        if _time_exceeded(deadline_s):
            break
    sol = evaluate_decoded_solution(inst, hubs, assign, tours_cust)
    return sol if sol.feasible else None


def move_swap(perm: List[int], i: int, j: int) -> List[int]:
    q = perm.copy()
    q[i], q[j] = q[j], q[i]
    return q


def move_insertion_before(perm: List[int], i: int, j: int) -> List[int]:
    q = perm.copy()
    v = q.pop(j)
    q.insert(i, v)
    return q



def ts_one_run(inst: dict, rng: np.random.Generator, time_limit: float, iter_mult: int, start_perm: List[int], fix_first_hub: bool) -> Tuple[Optional[HubSolution], dict]:
    n = inst["n"]
    max_iters = int(iter_mult * n)
    TL = np.zeros((n, n), dtype=int)
    t0 = _now_s()
    deadline_s = t0 + max(0.0, float(time_limit))
    cur_perm = start_perm.copy()
    cur_sol = decode_enhanced_feasible(inst, cur_perm, deadline_s=deadline_s)
    if cur_sol is None:
        return None, {
            "ok": False,
            "wall_s": float(_now_s() - t0),
            "max_iters": int(max_iters),
            "stopped_by_time": bool(_time_exceeded(deadline_s)),
        }
    best_sol = cur_sol
    improvement_history = [{"t_s": 0.0, "step": 0, "obj": float(best_sol.objective)}]
    first_feasible_obj = float(best_sol.objective)
    best_feasible_obj = float(best_sol.objective)
    time_to_first_feasible = 0.0
    time_to_best_feasible = 0.0
    first_feasible_step = 0
    best_feasible_step = 0
    it = 0
    stopped_by_time = False

    while it < max_iters:
        if _time_exceeded(deadline_s):
            stopped_by_time = True
            break
        it += 1
        TL = np.maximum(TL - 1, 0)
        newbest_sol = None
        newbest_perm = None
        newbest_pair = None
        start_i = 1 if fix_first_hub else 0
        time_up = False

        for i in range(start_i, n):
            if _time_exceeded(deadline_s):
                time_up = True
                break
            for j in range(max(i + 1, start_i), n):
                if _time_exceeded(deadline_s):
                    time_up = True
                    break
                if i >= j:
                    continue
                a = int(cur_perm[i])
                b = int(cur_perm[j])
                if TL[a, b] != 0 or TL[b, a] != 0:
                    continue
                for p2, pair in [(move_swap(cur_perm, i, j), (a, b)), (move_insertion_before(cur_perm, i, j), (a, b))]:
                    if _time_exceeded(deadline_s):
                        time_up = True
                        break
                    sol2 = decode_enhanced_feasible(inst, p2, deadline_s=deadline_s)
                    if sol2 is None:
                        if _time_exceeded(deadline_s):
                            time_up = True
                            break
                        continue
                    if (newbest_sol is None) or (sol2.objective < newbest_sol.objective - 1e-9):
                        newbest_sol = sol2
                        newbest_perm = p2
                        newbest_pair = pair
                if time_up:
                    break
            if time_up:
                break

        if time_up:
            stopped_by_time = True
            break

        if newbest_sol is None:
            if _time_exceeded(deadline_s):
                stopped_by_time = True
                break
            cur_perm = list(range(n))
            rng.shuffle(cur_perm)
            if fix_first_hub:
                h = start_perm[0]
                cur_perm = [h] + [x for x in cur_perm if x != h]
            TL[:] = 0
            cur_sol = decode_enhanced_feasible(inst, cur_perm, deadline_s=deadline_s)
            if cur_sol is None:
                if _time_exceeded(deadline_s):
                    stopped_by_time = True
                    break
                continue
        else:
            a, b = newbest_pair
            TL[a, b] = n
            TL[b, a] = n
            cur_perm = newbest_perm
            cur_sol = newbest_sol

        if cur_sol.objective < best_sol.objective - 1e-9:
            best_sol = cur_sol
            now_t = float(_now_s() - t0)
            best_feasible_obj = float(best_sol.objective)
            time_to_best_feasible = now_t
            best_feasible_step = it
            improvement_history.append({"t_s": now_t, "step": int(it), "obj": float(best_sol.objective)})

    if time_limit > 0 and _time_exceeded(deadline_s):
        stopped_by_time = True

    return best_sol, {
        "ok": True,
        "iters": int(it),
        "wall_s": float(_now_s() - t0),
        "max_iters": int(max_iters),
        "stopped_by_time": bool(stopped_by_time),
        "first_feasible_obj": first_feasible_obj,
        "best_feasible_obj": best_feasible_obj,
        "time_to_first_feasible_s": time_to_first_feasible,
        "time_to_best_feasible_s": time_to_best_feasible,
        "time_to_last_improvement_s": time_to_best_feasible,
        "first_feasible_step": first_feasible_step,
        "best_feasible_step": best_feasible_step,
        "last_improvement_step": best_feasible_step,
        "improvement_history": improvement_history,
    }


def run_multistart(inst: dict, seed: int, time_limit: float, iter_mult: int, restarts: int, hub_probe: bool, hub_probe_k: int, fix_first_hub: bool) -> Tuple[Optional[HubSolution], dict]:
    N = inst["N"]
    cand_hubs = list(np.argsort(np.array(inst["Fhub"], dtype=float)).astype(int).tolist())
    logs = []
    best_all = None
    t0 = _now_s()
    global_improvement_history: List[Dict[str, object]] = []
    first_feasible_obj = None
    best_feasible_obj = None
    time_to_first_feasible = None
    time_to_best_feasible = None
    first_feasible_step = None
    best_feasible_step = None
    total_step_offset = 0
    for r in range(max(1, int(restarts))):
        remaining = time_limit - (_now_s() - t0)
        if remaining <= 1e-9:
            break
        per_run = remaining / float(max(1, (restarts - r)))
        rng = np.random.default_rng(seed + 1000 * r)
        perm = N.copy()
        rng.shuffle(perm)
        forced = None
        fixed = False
        if hub_probe and r < max(0, int(hub_probe_k)):
            forced = cand_hubs[r % len(cand_hubs)]
            perm = [forced] + [x for x in perm if x != forced]
            fixed = fix_first_hub
        res_offset = float(_now_s() - t0)
        sol, st = ts_one_run(inst, rng, per_run, int(iter_mult), perm, fixed)
        if sol is None:
            if st.get("stopped_by_time", False):
                break
            continue
        st.update({"seed": int(seed + 1000 * r), "objective": float(sol.objective), "hubs": len(sol.hubs), "vehicles": sum(len(v) for v in sol.tours.values()), "first_hub_forced": int(forced) if forced is not None else None})
        logs.append(st)
        local_hist = st.get("improvement_history", [])
        for ev in local_hist:
            ge = {"t_s": float(ev["t_s"]) + res_offset, "step": int(ev["step"]) + total_step_offset, "obj": float(ev["obj"])}
            if first_feasible_obj is None:
                first_feasible_obj = ge["obj"]
                time_to_first_feasible = ge["t_s"]
                first_feasible_step = ge["step"]
            if best_feasible_obj is None or ge["obj"] < best_feasible_obj - 1e-9:
                best_feasible_obj = ge["obj"]
                time_to_best_feasible = ge["t_s"]
                best_feasible_step = ge["step"]
                global_improvement_history.append(ge)
        total_step_offset += int(st.get("iters", 0))
        if best_all is None or sol.objective < best_all.objective - 1e-9:
            best_all = sol
    total_wall = float(_now_s() - t0)
    stats = {
        "restarts": int(restarts),
        "used_runs": int(len(logs)),
        "total_wall_s": total_wall,
        "stopped_by_time": bool(time_limit > 0 and total_wall >= time_limit - 1e-6),
        "hub_probe": bool(hub_probe),
        "hub_probe_k": int(hub_probe_k),
        "fix_first_hub": bool(fix_first_hub),
        "cand_hubs_by_cost": cand_hubs[: min(10, len(cand_hubs))],
        "runs": logs,
        "first_feasible_obj": first_feasible_obj,
        "best_feasible_obj": best_feasible_obj,
        "time_to_first_feasible_s": time_to_first_feasible,
        "time_to_best_feasible_s": time_to_best_feasible,
        "time_to_last_improvement_s": time_to_best_feasible,
        "first_feasible_step": first_feasible_step,
        "best_feasible_step": best_feasible_step,
        "last_improvement_step": best_feasible_step,
        "improvement_history": global_improvement_history,
    }
    return best_all, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_json", type=str, default="")
    ap.add_argument("--data_dir", type=str, default="")
    ap.add_argument("--prefix", type=str, default="")
    ap.add_argument("--param_json", type=str, default="")
    ap.add_argument("--cap_type", type=str, default="L", choices=["L", "T", "l", "t"])
    ap.add_argument("--cost_type", type=str, default="L", choices=["L", "T", "l", "t"])
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--q", type=int, default=5)
    ap.add_argument("--F_vehicle", type=float, default=None)
    ap.add_argument("--assignment_weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--time_limit", type=float, default=1000.0)
    ap.add_argument("--iter_mult", type=int, default=20)
    ap.add_argument("--restarts", type=int, default=5)
    ap.add_argument("--hub_probe", type=int, default=1)
    ap.add_argument("--hub_probe_k", type=int, default=5)
    ap.add_argument("--fix_first_hub", type=int, default=1)
    ap.add_argument("--out_json", type=str, default="")
    args = ap.parse_args()

    if args.instance_json:
        inst = load_unified_json_instance(args.instance_json, q_override=None)
        if args.F_vehicle is not None:
            inst["F_vehicle"] = float(args.F_vehicle)
        if args.alpha is not None:
            inst["alpha"] = float(args.alpha)
    else:
        if not args.data_dir or not args.prefix:
            raise ValueError("either --instance_json or (--data_dir and --prefix) must be provided")
        inst = load_csv_param_instance(args.data_dir, args.prefix, args.param_json, args.cap_type, args.cost_type, float(args.alpha), int(args.q), args.F_vehicle, float(args.assignment_weight))

    sol, stats = run_multistart(inst, int(args.seed), float(args.time_limit), int(args.iter_mult), int(args.restarts), bool(args.hub_probe), int(args.hub_probe_k), bool(args.fix_first_hub))
    if sol is None:
        raise RuntimeError("TS failed to produce a feasible solution under the unified benchmark constraints")

    print(f"[TS-unified-v2] instance = {inst['name']}")
    print(f"[TS-unified-v2] objective = {sol.objective:.6f}")
    print(f"[TS-unified-v2] hubs = {sol.hubs}")
    print(f"[TS-unified-v2] vehicle_count = {sum(len(v) for v in sol.tours.values())}")
    print(f"[TS-unified-v2] costs: route={sol.route_cost:.6f}, assignment={sol.assignment_cost:.6f}, interhub={sol.interhub_cost:.6f}, vehicle={sol.vehicle_cost:.6f}, hub={sol.hub_cost:.6f}")

    if args.out_json:
        dataset_type = str(inst["name"]).split("_")[1] if "_" in str(inst["name"]) else None
        stop_reason = "time_limit" if bool(stats.get("stopped_by_time", False)) else "iteration_cap"
        runtime_utilization = None if float(args.time_limit) <= 0 else float(min(1.0, float(stats.get("total_wall_s", 0.0)) / max(float(args.time_limit), 1e-12)))
        out = {
            "instance": inst["name"],
            "dataset_type": dataset_type,
            "method": "TS",
            "seed": int(args.seed),
            "time_limit_s": float(args.time_limit),
            "total_wall_s": float(stats.get("total_wall_s", 0.0)),
            "stop_reason": stop_reason,
            "final_feasible": bool(sol.feasible),
            "final_obj": float(sol.objective) if sol.feasible else None,
            "source": inst["source"],
            "objective": sol.objective,
            "hubs": sol.hubs,
            "assign": {str(i): int(sol.assign[i]) for i in range(inst["n"])},
            "tours": {str(k): tours for k, tours in sol.tours.items()},
            "costs": {"route": sol.route_cost, "assignment": sol.assignment_cost, "interhub": sol.interhub_cost, "vehicle": sol.vehicle_cost, "hub": sol.hub_cost},
            "benchmark_params": {"alpha": float(inst["alpha"]), "q": int(inst["q"]), "F_vehicle": float(inst["F_vehicle"]), "assignment_weight": float(inst["assignment_weight"]), "Lambda": [float(x) for x in inst["Lambda"]], "Fhub": [float(x) for x in inst["Fhub"]]},
            "perf": {
                "algorithm": "TS",
                "seed": int(args.seed),
                "instance": inst["name"],
                "step_unit": "ts_iteration",
                "stopped_by_time": bool(stats.get("stopped_by_time", False)),
                "stop_reason": stop_reason,
                "time_limit_s": float(args.time_limit),
                "total_wall_s": float(stats.get("total_wall_s", 0.0)),
                "runtime_utilization": runtime_utilization,
                "first_feasible_obj": stats.get("first_feasible_obj"),
                "best_feasible_obj": stats.get("best_feasible_obj"),
                "time_to_first_feasible_s": stats.get("time_to_first_feasible_s"),
                "time_to_best_feasible_s": stats.get("time_to_best_feasible_s"),
                "time_to_last_improvement_s": stats.get("time_to_last_improvement_s"),
                "first_feasible_step": stats.get("first_feasible_step"),
                "best_feasible_step": stats.get("best_feasible_step"),
                "last_improvement_step": stats.get("last_improvement_step"),
                "num_improvements": int(len(stats.get("improvement_history", []))),
                "final_feasible": bool(sol.feasible),
            },
            "improvement_history": stats.get("improvement_history", []),
            "ts_settings": {"seed": int(args.seed), "time_limit": float(args.time_limit), "iter_mult": int(args.iter_mult), "restarts": int(args.restarts), "hub_probe": int(bool(args.hub_probe)), "hub_probe_k": int(args.hub_probe_k), "fix_first_hub": int(bool(args.fix_first_hub))},
            "ts_stats": stats,
            "feasible": bool(sol.feasible),
            "violations": list(sol.violations),
        }
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"[TS-unified-v2] wrote {args.out_json}")


if __name__ == "__main__":
    main()
