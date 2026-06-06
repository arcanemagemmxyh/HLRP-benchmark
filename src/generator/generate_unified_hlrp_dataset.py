# -*- coding: utf-8 -*-
r"""
generate_unified_hlrp_dataset_v2.py

Convert AP-derived CSV instances into a unified HLRP
benchmark JSON format for multi-algorithm comparison.

Unified benchmark v2
- single allocation
- hub fixed cost retained
- hub capacity retained
- local closed tours based on hubs
- route cardinality bound q
- inter-hub discount alpha
- assignment cost added in a mild, normalized form:
    assignment_weight * normalized_throughput[i] * distance(i, assigned_hub)
  where normalized_throughput[i] = (outbound_i + inbound_i) / mean_throughput
- costs kept as:
    hub fixed cost + local route cost + assignment cost + inter-hub cost + vehicle fixed cost
- Karimi-specific time-bound / covering constraints are NOT embedded here
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def load_edge_matrix(path: Path, value_col: str) -> np.ndarray:
    df = pd.read_csv(path)
    required = {"fromnode", "tonode", value_col}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} must contain columns {sorted(required)}")
    n = int(max(df["fromnode"].max(), df["tonode"].max()) + 1)
    mat = np.zeros((n, n), dtype=float)
    for row in df.itertuples(index=False):
        i = int(getattr(row, "fromnode"))
        j = int(getattr(row, "tonode"))
        mat[i, j] = float(getattr(row, value_col))
    return mat


def detect_prefixes(data_dir: Path) -> List[str]:
    prefixes = set()
    for p in data_dir.glob("*_nodes.csv"):
        prefixes.add(p.name[:-10])
    return sorted(prefixes)


def read_instance(data_dir: Path, prefix: str) -> Dict:
    nodes_path = data_dir / f"{prefix}_nodes.csv"
    c_path = data_dir / f"{prefix}_c.csv"
    w_path = data_dir / f"{prefix}_w.csv"
    params_path = data_dir / f"{prefix}_params_orlib.json"
    apmeta_path = data_dir / f"{prefix}_apmeta.json"

    missing = [str(p) for p in [nodes_path, c_path, w_path, params_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    nodes = pd.read_csv(nodes_path)
    if "ID" not in nodes.columns:
        raise ValueError(f"{nodes_path} must contain column 'ID'")
    if not {"latitude", "longitude"}.issubset(nodes.columns):
        raise ValueError(f"{nodes_path} must contain columns 'latitude' and 'longitude'")
    nodes = nodes.sort_values("ID").reset_index(drop=True)

    C = load_edge_matrix(c_path, "c")
    W = load_edge_matrix(w_path, "w")
    if C.shape != W.shape:
        raise ValueError(f"Cost and flow matrix shape mismatch for {prefix}: {C.shape} vs {W.shape}")

    with open(params_path, "r", encoding="utf-8") as f:
        params = json.load(f)

    apmeta = None
    if apmeta_path.exists():
        with open(apmeta_path, "r", encoding="utf-8") as f:
            apmeta = json.load(f)

    n = len(nodes)
    if C.shape != (n, n):
        raise ValueError(f"Node count mismatch for {prefix}: nodes={n}, matrix={C.shape}")

    return {
        "prefix": prefix,
        "n": n,
        "nodes": nodes,
        "C": C,
        "W": W,
        "params": params,
        "apmeta": apmeta,
    }


def build_unified_instance(
    raw: Dict,
    cap_type: str,
    cost_type: str,
    alpha: float,
    q: int,
    vehicle_fixed_cost: float,
    keep_diag_flow: bool,
    assignment_weight: float,
) -> Dict:
    cap_type = cap_type.upper()
    cost_type = cost_type.upper()
    if cap_type not in {"L", "T"}:
        raise ValueError("cap_type must be L or T")
    if cost_type not in {"L", "T"}:
        raise ValueError("cost_type must be L or T")

    n = raw["n"]
    nodes = raw["nodes"]
    C = raw["C"].copy()
    W = raw["W"].copy()
    params = raw["params"]
    apmeta = raw["apmeta"] or {}

    if not keep_diag_flow:
        np.fill_diagonal(W, 0.0)

    f_key = f"Fhub_{cost_type}"
    l_key = f"Lambda_{cap_type}"
    if f_key not in params or l_key not in params:
        raise KeyError(f"{raw['prefix']}_params_orlib.json must contain '{f_key}' and '{l_key}'")

    Fhub = np.array(params[f_key], dtype=float).reshape(-1)
    Lambda = np.array(params[l_key], dtype=float).reshape(-1)
    if len(Fhub) != n or len(Lambda) != n:
        raise ValueError(
            f"Parameter length mismatch for {raw['prefix']}: len(Fhub)={len(Fhub)}, len(Lambda)={len(Lambda)}, n={n}"
        )

    outbound = W.sum(axis=1)
    inbound = W.sum(axis=0)
    throughput = outbound + inbound
    mean_throughput = float(np.mean(throughput)) if n > 0 else 1.0
    if mean_throughput <= 1e-12:
        mean_throughput = 1.0
    throughput_norm = throughput / mean_throughput
    total_flow = float(W.sum())

    coord_list = []
    for row in nodes.itertuples(index=False):
        coord_list.append({
            "id": int(row.ID),
            "x": float(row.latitude),
            "y": float(row.longitude),
        })

    distance_edges = []
    flow_edges = []
    for i in range(n):
        for j in range(n):
            distance_edges.append({"from": i, "to": j, "cost": float(C[i, j])})
            if abs(W[i, j]) > 1e-12:
                flow_edges.append({"from": i, "to": j, "flow": float(W[i, j])})

    instance_name = f"{raw['prefix']}_{cost_type}{cap_type}_q{q}"
    out = {
        "instance_name": instance_name,
        "source_prefix": raw["prefix"],
        "benchmark_version": "unified_hlrp_v2",
        "modeling_choices": {
            "single_allocation": True,
            "use_hub_capacity": True,
            "use_hub_fixed_cost": True,
            "use_vehicle_capacity": False,
            "use_karimi_covering_timebound": False,
            "local_tours": "closed tours starting and ending at the same hub",
            "route_cardinality_bound_q": int(q),
            "candidate_hubs": "all nodes",
            "assignment_cost_mode": "normalized_throughput_distance",
            "objective_terms": [
                "hub_fixed_cost",
                "local_route_cost",
                "assignment_cost",
                "inter_hub_flow_cost",
                "vehicle_fixed_cost"
            ]
        },
        "parameters": {
            "alpha": float(alpha),
            "cap_type": cap_type,
            "cost_type": cost_type,
            "q": int(q),
            "vehicle_fixed_cost": float(vehicle_fixed_cost),
            "keep_diag_flow": bool(keep_diag_flow),
            "assignment_weight": float(assignment_weight),
            "throughput_normalization_mean": float(mean_throughput),
            "ap_meta_collect": apmeta.get("collect"),
            "ap_meta_transfer": apmeta.get("transfer"),
            "ap_meta_distribute": apmeta.get("distribute"),
            "ap_meta_p": apmeta.get("p"),
        },
        "summary": {
            "n_nodes": int(n),
            "n_candidate_hubs": int(n),
            "total_flow": total_flow,
            "total_outbound": float(outbound.sum()),
            "total_inbound": float(inbound.sum()),
            "avg_nonzero_distance": float(np.mean(C[C > 0])) if np.any(C > 0) else 0.0,
            "avg_nonzero_flow": float(np.mean(W[W > 0])) if np.any(W > 0) else 0.0,
            "avg_throughput": float(np.mean(throughput)),
            "min_hub_fixed_cost": float(Fhub.min()),
            "max_hub_fixed_cost": float(Fhub.max()),
            "min_hub_capacity": float(Lambda.min()),
            "max_hub_capacity": float(Lambda.max()),
        },
        "nodes": coord_list,
        "hub_fixed_cost": [float(x) for x in Fhub.tolist()],
        "hub_capacity": [float(x) for x in Lambda.tolist()],
        "outbound_flow": [float(x) for x in outbound.tolist()],
        "inbound_flow": [float(x) for x in inbound.tolist()],
        "node_throughput": [float(x) for x in throughput.tolist()],
        "node_throughput_normalized": [float(x) for x in throughput_norm.tolist()],
        "distance_matrix": [[float(x) for x in row] for row in C.tolist()],
        "flow_matrix": [[float(x) for x in row] for row in W.tolist()],
        "distance_edges": distance_edges,
        "flow_edges": flow_edges,
    }
    return out


def write_manifest(rows: List[Dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "instance_name", "source_prefix", "n_nodes", "cap_type", "cost_type", "q", "alpha",
        "assignment_weight", "total_flow", "min_hub_fixed_cost", "max_hub_fixed_cost",
        "min_hub_capacity", "max_hub_capacity", "json_path"
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate unified HLRP benchmark JSON files from Karimi AP-derived data.")
    parser.add_argument("--data_dir", type=str, required=True, help="Input folder containing AP-derived CSV files")
    parser.add_argument("--out_dir", type=str, required=True, help="Output folder, e.g. ./TS/UnifiedData")
    parser.add_argument("--prefixes", type=str, default="", help="Comma-separated prefixes, e.g. ap10,ap20,ap25. Leave empty to auto-detect.")
    parser.add_argument("--cap_type", type=str, default="L", choices=["L", "T", "l", "t"], help="Hub capacity type")
    parser.add_argument("--cost_type", type=str, default="L", choices=["L", "T", "l", "t"], help="Hub fixed-cost type")
    parser.add_argument("--alpha", type=float, default=0.75, help="Inter-hub discount used by the unified benchmark")
    parser.add_argument("--q", type=int, default=5, help="Max customers per local route")
    parser.add_argument("--vehicle_fixed_cost", type=float, default=0.0, help="Vehicle fixed cost to store in benchmark metadata")
    parser.add_argument("--assignment_weight", type=float, default=1.0, help="Weight of normalized throughput-distance assignment cost")
    parser.add_argument("--keep_diag_flow", type=int, default=0, choices=[0, 1], help="Keep w[i,i] if set to 1, otherwise set diagonal flow to zero")
    parser.add_argument("--pretty", type=int, default=1, choices=[0, 1], help="Pretty-print JSON")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.prefixes.strip():
        prefixes = [x.strip() for x in args.prefixes.split(",") if x.strip()]
    else:
        prefixes = detect_prefixes(data_dir)

    if not prefixes:
        raise ValueError(f"No prefixes detected under {data_dir}")

    manifest_rows = []
    for prefix in prefixes:
        raw = read_instance(data_dir, prefix)
        unified = build_unified_instance(
            raw=raw,
            cap_type=args.cap_type,
            cost_type=args.cost_type,
            alpha=args.alpha,
            q=args.q,
            vehicle_fixed_cost=args.vehicle_fixed_cost,
            keep_diag_flow=bool(args.keep_diag_flow),
            assignment_weight=float(args.assignment_weight),
        )
        out_name = f"{unified['instance_name']}.json"
        out_path = out_dir / out_name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(unified, f, ensure_ascii=False, indent=2 if args.pretty else None)

        summary = unified["summary"]
        manifest_rows.append({
            "instance_name": unified["instance_name"],
            "source_prefix": unified["source_prefix"],
            "n_nodes": summary["n_nodes"],
            "cap_type": unified["parameters"]["cap_type"],
            "cost_type": unified["parameters"]["cost_type"],
            "q": unified["parameters"]["q"],
            "alpha": unified["parameters"]["alpha"],
            "assignment_weight": unified["parameters"]["assignment_weight"],
            "total_flow": summary["total_flow"],
            "min_hub_fixed_cost": summary["min_hub_fixed_cost"],
            "max_hub_fixed_cost": summary["max_hub_fixed_cost"],
            "min_hub_capacity": summary["min_hub_capacity"],
            "max_hub_capacity": summary["max_hub_capacity"],
            "json_path": str(out_path),
        })
        print(f"[OK] {prefix} -> {out_path}")

    manifest_path = out_dir / "manifest.csv"
    write_manifest(manifest_rows, manifest_path)
    print(f"[DONE] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
