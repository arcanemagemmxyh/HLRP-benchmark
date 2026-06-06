# -*- coding: utf-8 -*-
"""
Prepare AP/CAB/TR instances in the unified HLRP JSON format.

The script is intended for topology-transfer and scalability experiments. AP
instances exported from OR-Library keep their original hub fixed costs and hub
capacities when the corresponding params file is available. Sources without
HLRP-specific hub parameters, such as CAB/TR topology-transfer branches, receive
source-neutral transformed parameters derived from throughput and distance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "src" / "generator"
EVALUATOR = ROOT / "src" / "evaluator"
for path in (GENERATOR, EVALUATOR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from generate_unified_hlrp_dataset import build_unified_instance, read_instance  # noqa: E402
from hlrp_instance import HLRPInstance  # noqa: E402


def _load_edge_matrix(path: Path, value_col: str) -> np.ndarray:
    df = pd.read_csv(path)
    n = int(max(df["fromnode"].max(), df["tonode"].max()) + 1)
    mat = np.zeros((n, n), dtype=float)
    for row in df.itertuples(index=False):
        mat[int(row.fromnode), int(row.tonode)] = float(getattr(row, value_col))
    return mat


def _write_edge_matrix(path: Path, mat: np.ndarray, value_col: str) -> None:
    rows = []
    n = mat.shape[0]
    for i in range(n):
        for j in range(n):
            rows.append((i, j, float(mat[i, j])))
    pd.DataFrame(rows, columns=["fromnode", "tonode", value_col]).to_csv(path, index=False)


def _euclidean_cost(coords: np.ndarray) -> np.ndarray:
    n = coords.shape[0]
    cost = np.zeros((n, n), dtype=float)
    for i in range(n):
        xi, yi = coords[i]
        for j in range(n):
            xj, yj = coords[j]
            cost[i, j] = math.hypot(xi - xj, yi - yj)
    return cost


def _classical_mds(distance: np.ndarray) -> np.ndarray:
    """Build deterministic 2D display coordinates from a distance matrix."""
    n = distance.shape[0]
    if n == 0:
        return np.zeros((0, 2), dtype=float)
    d2 = np.square(distance.astype(float))
    j = np.eye(n) - np.ones((n, n), dtype=float) / n
    b = -0.5 * j @ d2 @ j
    vals, vecs = np.linalg.eigh(b)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    coords = np.zeros((n, 2), dtype=float)
    for k in range(min(2, n)):
        coords[:, k] = vecs[:, k] * math.sqrt(max(float(vals[k]), 0.0))
    return coords


def _numbers(text: str) -> List[float]:
    vals: List[float] = []
    for tok in text.replace(",", " ").split():
        try:
            vals.append(float(tok))
        except ValueError:
            pass
    return vals


def export_cab_from_orlib_zip(zip_path: Path, out_dir: Path, prefix: str, n: int) -> None:
    """Export official CAB n.3.4 file to nodes/c/w CSV files."""
    member = f"CAB/{n}.3.4"
    with zipfile.ZipFile(zip_path, "r") as zf:
        lines = zf.read(member).decode("utf-8", errors="ignore").splitlines()
    lines = [ln.strip() for ln in lines if ln.strip()]
    n_file = int(float(lines[0]))
    if n_file != n:
        raise ValueError(f"{member} contains n={n_file}, expected {n}")

    coords = []
    for line in lines[1: 1 + n]:
        x, y = line.split()[:2]
        coords.append((float(x), float(y)))
    coords_arr = np.asarray(coords, dtype=float)

    flow_tokens: List[float] = []
    for line in lines[1 + n: 1 + n + n]:
        flow_tokens.extend(_numbers(line))
    if len(flow_tokens) < n * n:
        raise ValueError(f"{member} has only {len(flow_tokens)} flow tokens")
    W = np.asarray(flow_tokens[: n * n], dtype=float).reshape((n, n))
    C = _euclidean_cost(coords_arr)

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "ID": np.arange(n, dtype=int),
        "latitude": coords_arr[:, 0],
        "longitude": coords_arr[:, 1],
    }).to_csv(out_dir / f"{prefix}_nodes.csv", index=False)
    _write_edge_matrix(out_dir / f"{prefix}_c.csv", C, "c")
    _write_edge_matrix(out_dir / f"{prefix}_w.csv", W, "w")


def export_tr_from_xls(xls_path: Path, out_dir: Path, prefix: str, n: int = 81) -> None:
    """Export the 81-node Turkish network workbook to nodes/c/w CSV files."""
    if n != 81:
        raise ValueError("The official Turkish network workbook contains 81 nodes.")
    try:
        import win32com.client as win32  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Reading Turkish_network.xls requires pywin32/Excel COM on Windows.") from exc

    excel = win32.Dispatch("Excel.Application")
    excel.Visible = False
    excel.DisplayAlerts = False
    wb = None
    try:
        wb = excel.Workbooks.Open(str(xls_path.resolve()))
        ws_d = wb.Worksheets("Distance (km)")
        ws_w = wb.Worksheets("Flow")
        ws_f = wb.Worksheets("Fixed hub cost")

        city_names: List[str] = []
        D = np.zeros((n, n), dtype=float)
        W = np.zeros((n, n), dtype=float)
        fixed_hub_cost = np.zeros(n, dtype=float)

        for i in range(n):
            city_names.append(str(ws_d.Cells(i + 3, 2).Value))
            for j in range(n):
                D[i, j] = float(ws_d.Cells(i + 3, j + 3).Value or 0.0)
                W[i, j] = float(ws_w.Cells(i + 3, j + 3).Value or 0.0)
            fixed_hub_cost[i] = float(ws_f.Cells(i + 2, 3).Value or 0.0)
    finally:
        if wb is not None:
            wb.Close(False)
        excel.Quit()

    coords = _classical_mds(D)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "ID": np.arange(n, dtype=int),
        "latitude": coords[:, 0],
        "longitude": coords[:, 1],
        "city": city_names,
    }).to_csv(out_dir / f"{prefix}_nodes.csv", index=False, encoding="utf-8")
    _write_edge_matrix(out_dir / f"{prefix}_c.csv", D, "c")
    _write_edge_matrix(out_dir / f"{prefix}_w.csv", W, "w")
    meta = {
        "source": "Turkish network.xls",
        "n": n,
        "city_names": city_names,
        "raw_fixed_hub_cost": [float(x) for x in fixed_hub_cost],
        "coordinate_note": "Display coordinates are generated by classical MDS from the distance matrix.",
    }
    (out_dir / f"{prefix}_apmeta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_orlib_vector(zf: zipfile.ZipFile, member: str, n: int) -> np.ndarray:
    vals = _numbers(zf.read(member).decode("utf-8", errors="ignore"))
    if len(vals) == n + 1 and int(round(vals[0])) == n:
        vals = vals[1:]
    if len(vals) < n:
        raise ValueError(f"{member} has {len(vals)} values, expected at least {n}")
    return np.asarray(vals[:n], dtype=float)


def export_ap_from_orlib_zip(zip_path: Path, out_dir: Path, prefix: str, n: int, base_n: Optional[int] = None) -> None:
    """Export official AP n.3 file, or a deterministic first-n subset of a larger AP file."""
    source_n = int(base_n or n)
    member = f"AP/{source_n}.3"
    with zipfile.ZipFile(zip_path, "r") as zf:
        lines = zf.read(member).decode("utf-8", errors="ignore").splitlines()
        lines = [ln.strip() for ln in lines if ln.strip()]
        n_file = int(float(lines[0]))
        if n_file != source_n:
            raise ValueError(f"{member} contains n={n_file}, expected {source_n}")
        if n > source_n:
            raise ValueError(f"Cannot export first {n} nodes from AP/{source_n}.3")

        coords = []
        for line in lines[1: 1 + source_n]:
            x, y = line.split()[:2]
            coords.append((float(x), float(y)))
        coords_arr = np.asarray(coords, dtype=float)[:n, :]

        flow_tokens: List[float] = []
        for line in lines[1 + source_n: 1 + source_n + source_n]:
            flow_tokens.extend(_numbers(line))
        if len(flow_tokens) < source_n * source_n:
            raise ValueError(f"{member} has only {len(flow_tokens)} flow tokens")
        W_full = np.asarray(flow_tokens[: source_n * source_n], dtype=float).reshape((source_n, source_n))
        W = W_full[:n, :n]
        C = _euclidean_cost(coords_arr)

        tail = lines[1 + source_n + source_n:]
        meta = {
            "n": n,
            "p": int(float(tail[0])) if len(tail) >= 1 else None,
            "collect": float(tail[1]) if len(tail) >= 2 else None,
            "transfer": float(tail[2]) if len(tail) >= 3 else None,
            "distribute": float(tail[3]) if len(tail) >= 4 else None,
            "source_n": source_n,
        }
        params = {
            "Fhub_L": _read_orlib_vector(zf, f"AP/FcostL.{source_n}", source_n)[:n].tolist(),
            "Fhub_T": _read_orlib_vector(zf, f"AP/FcostT.{source_n}", source_n)[:n].tolist(),
            "Lambda_L": _read_orlib_vector(zf, f"AP/CapL.{source_n}", source_n)[:n].tolist(),
            "Lambda_T": _read_orlib_vector(zf, f"AP/CapT.{source_n}", source_n)[:n].tolist(),
            "meta": {
                "source": "APdata.zip",
                "parameter_rule": "orlib_original" if n == source_n else "orlib_first_n_subset",
                "inst": f"{source_n}.3",
                "n": n,
                "source_n": source_n,
            },
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "ID": np.arange(n, dtype=int),
        "latitude": coords_arr[:, 0],
        "longitude": coords_arr[:, 1],
    }).to_csv(out_dir / f"{prefix}_nodes.csv", index=False)
    _write_edge_matrix(out_dir / f"{prefix}_c.csv", C, "c")
    _write_edge_matrix(out_dir / f"{prefix}_w.csv", W, "w")
    (out_dir / f"{prefix}_params_orlib.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    meta["note"] = "Costs computed as Euclidean distance. collect/transfer/distribute preserved from AP instance file."
    (out_dir / f"{prefix}_apmeta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def copy_csv_source(src_dir: Path, out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ["nodes", "c", "w"]:
        src = src_dir / f"{prefix}_{suffix}.csv"
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copyfile(src, out_dir / src.name)
    for suffix in ["params_orlib", "apmeta"]:
        src = src_dir / f"{prefix}_{suffix}.json"
        if src.exists():
            shutil.copyfile(src, out_dir / src.name)


def write_transformed_params(stage_dir: Path, prefix: str, source: str) -> Path:
    C = _load_edge_matrix(stage_dir / f"{prefix}_c.csv", "c")
    W = _load_edge_matrix(stage_dir / f"{prefix}_w.csv", "w")
    np.fill_diagonal(W, 0.0)
    n = C.shape[0]

    outbound = W.sum(axis=1)
    inbound = W.sum(axis=0)
    throughput = outbound + inbound
    total_throughput = float(throughput.sum())
    mean_throughput = float(np.mean(throughput)) if n else 1.0
    if mean_throughput <= 1e-12:
        mean_throughput = 1.0

    weighted_distance = float((W * C).sum() / max(W.sum(), 1.0))
    if weighted_distance <= 1e-12:
        weighted_distance = float(np.mean(C[C > 0])) if np.any(C > 0) else 1.0

    tp_min = float(throughput.min()) if n else 0.0
    tp_range = float(throughput.max() - throughput.min()) if n else 0.0
    if tp_range <= 1e-12:
        tp_score = np.ones(n, dtype=float) * 0.5
    else:
        tp_score = (throughput - tp_min) / tp_range

    positive = np.where(C > 0, C, np.nan)
    avg_dist = np.nanmean(positive, axis=1)
    avg_dist = np.where(np.isfinite(avg_dist), avg_dist, np.nanmean(positive))
    centrality = 1.0 / np.maximum(avg_dist, 1e-9)
    cent_range = float(centrality.max() - centrality.min()) if n else 0.0
    if cent_range <= 1e-12:
        cent_score = np.ones(n, dtype=float) * 0.5
    else:
        cent_score = (centrality - centrality.min()) / cent_range

    target_hubs = max(2, int(round(math.sqrt(max(n, 1)))))
    capacity_base = total_throughput / target_hubs if target_hubs else total_throughput
    Lambda_L = np.maximum(1.05 * throughput, capacity_base * (0.80 + 0.60 * tp_score))
    Lambda_T = np.maximum(1.02 * throughput, capacity_base * (0.55 + 0.45 * tp_score))

    fixed_base = 0.003 * weighted_distance * mean_throughput
    Fhub_L = fixed_base * (0.85 + 0.30 * tp_score + 0.15 * (1.0 - cent_score))
    Fhub_T = 1.25 * Fhub_L

    params = {
        "Fhub_L": [float(x) for x in Fhub_L],
        "Fhub_T": [float(x) for x in Fhub_T],
        "Lambda_L": [float(x) for x in Lambda_L],
        "Lambda_T": [float(x) for x in Lambda_T],
        "meta": {
            "source": source,
            "parameter_rule": "source_neutral_throughput_distance_v1",
            "note": (
                "Hub opening costs and capacities are transformed from throughput "
                "and distance because the source topology does not provide HLRP "
                "fixed-cost and capacity parameters."
            ),
            "target_hubs": target_hubs,
            "weighted_distance": weighted_distance,
            "mean_throughput": mean_throughput,
        },
    }
    out = stage_dir / f"{prefix}_params_orlib.json"
    out.write_text(json.dumps(params, indent=2), encoding="utf-8")
    return out


def write_unified(
    stage_dir: Path,
    out_root: Path,
    prefix: str,
    setting: str,
    q: int,
    alpha: float,
    assignment_weight: float,
    vehicle_fixed_cost: float,
) -> Path:
    setting = setting.upper()
    if setting not in {"LL", "TT"}:
        raise ValueError("This script currently prepares LL or TT settings.")
    cost_type, cap_type = setting[0], setting[1]
    raw = read_instance(stage_dir, prefix)
    unified = build_unified_instance(
        raw=raw,
        cap_type=cap_type,
        cost_type=cost_type,
        alpha=alpha,
        q=q,
        vehicle_fixed_cost=vehicle_fixed_cost,
        keep_diag_flow=False,
        assignment_weight=assignment_weight,
    )
    params_meta = raw["params"].get("meta", {}) if isinstance(raw["params"], dict) else {}
    unified.setdefault("parameters", {})["source_parameter_rule"] = params_meta.get("parameter_rule", "orlib_original")
    unified.setdefault("parameters", {})["source_parameter_note"] = params_meta.get("note")
    out_dir = out_root / setting
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{unified['instance_name']}.json"
    out.write_text(json.dumps(unified, indent=2), encoding="utf-8")
    return out


def smoke_solution(n: int, q: int) -> Dict:
    open_hubs = list(range(n))
    return {
        "open_hubs": open_hubs,
        "assignment": {str(i): i for i in range(n)},
        "routes": {str(i): [] for i in range(n)},
    }


def write_manifest(rows: List[Dict], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare cross-topology HLRP JSON instances.")
    parser.add_argument("--local_data_dir", type=Path, default=ROOT / "data" / "source_csv")
    parser.add_argument("--ap_data_dir", type=Path, default=ROOT / "data" / "source_ap")
    parser.add_argument("--ap_zip", type=Path, default=ROOT / "data" / "raw_sources" / "APdata.zip")
    parser.add_argument("--cab_zip", type=Path, default=ROOT / "data" / "raw_sources" / "CABdata.zip")
    parser.add_argument("--tr_xls", type=Path, default=ROOT / "data" / "raw_sources" / "Turkish_network.xls")
    parser.add_argument("--stage_dir", type=Path, default=ROOT / "scratch" / "stage_cross_topology")
    parser.add_argument("--out_dir", type=Path, default=ROOT / "data" / "cross_source")
    parser.add_argument("--prefixes", type=str, default="ap10,cab10,tr10")
    parser.add_argument("--settings", type=str, default="LL,TT")
    parser.add_argument("--q", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--assignment_weight", type=float, default=1.0)
    parser.add_argument("--vehicle_fixed_cost", type=float, default=0.0)
    parser.add_argument("--force_transformed_params", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefixes = [x.strip() for x in args.prefixes.split(",") if x.strip()]
    settings = [x.strip().upper() for x in args.settings.split(",") if x.strip()]
    rows: List[Dict] = []

    for prefix in prefixes:
        source = "".join(ch for ch in prefix if not ch.isdigit()).lower()
        n_digits = "".join(ch for ch in prefix if ch.isdigit())
        n = int(n_digits) if n_digits else None
        stage = args.stage_dir / prefix
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True, exist_ok=True)

        if source == "ap" and args.ap_zip.exists() and n in {100, 200}:
            export_ap_from_orlib_zip(args.ap_zip, stage, prefix, n)
            parameter_source = "APdata.zip"
        elif source == "ap" and args.ap_zip.exists() and n == 90:
            export_ap_from_orlib_zip(args.ap_zip, stage, prefix, n, base_n=100)
            parameter_source = "APdata.zip"
        elif source == "cab" and args.cab_zip.exists() and n in {10, 15, 20, 25}:
            export_cab_from_orlib_zip(args.cab_zip, stage, prefix, n)
            parameter_source = "CABdata.zip"
        elif source == "tr" and args.tr_xls.exists() and n == 81:
            export_tr_from_xls(args.tr_xls, stage, prefix, n)
            parameter_source = "Turkish network.xls"
        elif source == "ap" and (args.ap_data_dir / f"{prefix}_nodes.csv").exists():
            copy_csv_source(args.ap_data_dir, stage, prefix)
            parameter_source = "APdata.zip"
        else:
            copy_csv_source(args.local_data_dir, stage, prefix)
            parameter_source = f"local_csv:{source}"

        params_path = stage / f"{prefix}_params_orlib.json"
        if args.force_transformed_params or not params_path.exists():
            write_transformed_params(stage, prefix, parameter_source)

        for setting in settings:
            out = write_unified(
                stage_dir=stage,
                out_root=args.out_dir,
                prefix=prefix,
                setting=setting,
                q=args.q,
                alpha=args.alpha,
                assignment_weight=args.assignment_weight,
                vehicle_fixed_cost=args.vehicle_fixed_cost,
            )
            data = json.loads(out.read_text(encoding="utf-8"))
            inst = HLRPInstance(data)
            ev = inst.evaluate_solution(smoke_solution(inst.n, inst.q), raise_on_invalid=False)
            if not ev.feasible:
                raise RuntimeError(f"Smoke solution infeasible for {out}: {ev.violations}")
            rows.append({
                "instance_name": data["instance_name"],
                "source_prefix": data["source_prefix"],
                "setting": setting,
                "n_nodes": data["summary"]["n_nodes"],
                "q": data["parameters"]["q"],
                "alpha": data["parameters"]["alpha"],
                "parameter_rule": data["parameters"].get("source_parameter_rule"),
                "total_flow": data["summary"]["total_flow"],
                "min_capacity": data["summary"]["min_hub_capacity"],
                "max_capacity": data["summary"]["max_hub_capacity"],
                "min_fixed_cost": data["summary"]["min_hub_fixed_cost"],
                "max_fixed_cost": data["summary"]["max_hub_fixed_cost"],
                "all_hub_smoke_cost": ev.total_cost,
                "json_path": str(out),
            })
            print(f"[OK] {out} smoke_cost={ev.total_cost:.3f}")

    manifest = args.out_dir / "manifest.csv"
    write_manifest(rows, manifest)
    print(f"[DONE] manifest -> {manifest}")


if __name__ == "__main__":
    main()
