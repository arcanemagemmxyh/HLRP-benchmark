# -*- coding: utf-8 -*-
"""Generate the dense small-certified HLRP benchmark grid.

Default grid:
    scale  : ap10, ap15
    regime : LL, TT
    q      : 3, 4, 5
    alpha  : 0.2, 0.3, ..., 0.9

Total: 2 x 2 x 3 x 8 = 96 instances.

ap15 is constructed as a deterministic AP-derived subset: the first 15 nodes of
ap20, together with the corresponding first 15 cost/capacity entries. This keeps
all AP-derived instances reproducible while adding a denser small-certified tier.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "src" / "generator"
for path in (GENERATOR, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from generate_unified_hlrp_dataset import build_unified_instance, read_instance

DEFAULT_SCALES = ("ap10", "ap15")
DEFAULT_REGIMES = ("LL", "TT")
DEFAULT_Q_VALUES = (3, 4, 5)
DEFAULT_ALPHAS = tuple(round(0.1 * i, 1) for i in range(2, 10))


def alpha_tag(alpha: float) -> str:
    return f"a{int(round(float(alpha) * 100)):03d}"


def parse_alphas(values: Sequence[str]) -> List[float]:
    if len(values) == 1 and values[0].lower() in {"wandelt", "default", "02_09"}:
        return list(DEFAULT_ALPHAS)
    return [round(float(x), 10) for x in values]


def subset_raw(raw: Dict, prefix: str, n_keep: int) -> Dict:
    """Return a first-n AP-derived subset while preserving parameter semantics."""
    nodes = raw["nodes"].iloc[:n_keep].copy().reset_index(drop=True)
    nodes["ID"] = range(n_keep)

    params = {}
    for key, value in raw["params"].items():
        if isinstance(value, list):
            params[key] = value[:n_keep]
        else:
            params[key] = value
    if isinstance(params.get("meta"), dict):
        params["meta"] = dict(params["meta"])
        params["meta"]["source_prefix"] = raw["prefix"]
        params["meta"]["subset_first_n"] = n_keep

    apmeta = dict(raw.get("apmeta") or {})
    apmeta["source_prefix"] = raw["prefix"]
    apmeta["subset_first_n"] = n_keep
    apmeta["note"] = "AP-derived first-n subset for the dense small-certified HLRP grid."

    return {
        "prefix": prefix,
        "n": n_keep,
        "nodes": nodes,
        "C": raw["C"][:n_keep, :n_keep].copy(),
        "W": raw["W"][:n_keep, :n_keep].copy(),
        "params": params,
        "apmeta": apmeta,
    }


def load_scale(data_dir: Path, scale: str) -> Dict:
    if scale == "ap15":
        return subset_raw(read_instance(data_dir, "ap20"), "ap15", 15)
    return read_instance(data_dir, scale)


def iter_grid(scales: Sequence[str], regimes: Sequence[str], q_values: Sequence[int], alphas: Sequence[float]) -> Iterable[tuple[str, str, int, float]]:
    for scale in scales:
        for regime in regimes:
            for q in q_values:
                for alpha in alphas:
                    yield scale, regime, int(q), float(alpha)


def write_manifest(rows: List[Dict], manifest_path: Path) -> None:
    fields = [
        "instance_name",
        "scale",
        "regime",
        "q",
        "alpha",
        "alpha_tag",
        "n_nodes",
        "total_flow",
        "min_hub_fixed_cost",
        "max_hub_fixed_cost",
        "min_hub_capacity",
        "max_hub_capacity",
        "json_path",
    ]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dense small-certified HLRP instances.")
    parser.add_argument("--data_dir", type=str, default="data/source_ap", help="Directory containing AP-derived source files.")
    parser.add_argument("--out_dir", type=str, default="data/small_certified", help="Output directory for generated JSON instances.")
    parser.add_argument("--scales", nargs="+", default=list(DEFAULT_SCALES), help="Scale names, e.g., ap10 ap15.")
    parser.add_argument("--regimes", nargs="+", default=list(DEFAULT_REGIMES), help="Regimes, e.g., LL TT.")
    parser.add_argument("--q_values", nargs="+", type=int, default=list(DEFAULT_Q_VALUES), help="Route-size limits.")
    parser.add_argument("--alphas", nargs="+", default=["default"], help="Alpha values. Use default for 0.2,...,0.9.")
    parser.add_argument("--vehicle_fixed_cost", type=float, default=0.0)
    parser.add_argument("--assignment_weight", type=float, default=1.0)
    parser.add_argument("--pretty", type=int, default=1, choices=[0, 1])
    parser.add_argument("--overwrite", type=int, default=1, choices=[0, 1])
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    alphas = parse_alphas(args.alphas)
    raw_cache = {scale: load_scale(data_dir, scale) for scale in args.scales}
    rows: List[Dict] = []

    for scale, regime, q, alpha in iter_grid(args.scales, args.regimes, args.q_values, alphas):
        if len(regime) != 2 or regime[0] not in {"L", "T"} or regime[1] not in {"L", "T"}:
            raise ValueError(f"Invalid regime: {regime}. Expected one of LL, LT, TL, TT.")
        cost_type = regime[0]
        cap_type = regime[1]
        raw = raw_cache[scale]
        inst = build_unified_instance(
            raw=raw,
            cap_type=cap_type,
            cost_type=cost_type,
            alpha=alpha,
            q=q,
            vehicle_fixed_cost=args.vehicle_fixed_cost,
            keep_diag_flow=False,
            assignment_weight=args.assignment_weight,
        )

        tag = alpha_tag(alpha)
        instance_name = f"{scale}_{regime}_q{q}_{tag}"
        inst["instance_name"] = instance_name
        inst["source_prefix"] = scale
        inst.setdefault("parameters", {})["alpha_tag"] = tag
        inst.setdefault("parameters", {})["small_certified_grid"] = "ap10/ap15 x LL/TT x q=3/4/5 x alpha=0.2,...,0.9"
        inst["small_certified"] = {
            "scale": scale,
            "regime": regime,
            "q": q,
            "alpha": alpha,
            "alpha_tag": tag,
            "fixed_cost_scale": 1.0,
            "perturbation_seed": None,
            "ap15_construction": "first 15 nodes of ap20" if scale == "ap15" else None,
        }

        regime_dir = out_dir / regime
        regime_dir.mkdir(parents=True, exist_ok=True)
        out_path = regime_dir / f"{instance_name}.json"
        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] {instance_name} already exists")
        else:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(inst, f, ensure_ascii=False, indent=2 if args.pretty else None)
            print(f"[OK] {instance_name} -> {out_path}")

        summary = inst["summary"]
        rows.append({
            "instance_name": instance_name,
            "scale": scale,
            "regime": regime,
            "q": q,
            "alpha": alpha,
            "alpha_tag": tag,
            "n_nodes": summary["n_nodes"],
            "total_flow": summary["total_flow"],
            "min_hub_fixed_cost": summary["min_hub_fixed_cost"],
            "max_hub_fixed_cost": summary["max_hub_fixed_cost"],
            "min_hub_capacity": summary["min_hub_capacity"],
            "max_hub_capacity": summary["max_hub_capacity"],
            "json_path": str(out_path).replace("/", "\\"),
        })

    write_manifest(rows, out_dir / "manifest.csv")
    print(f"[DONE] generated {len(rows)} manifest rows")
    print(f"[DONE] manifest -> {out_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
