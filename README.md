# HLRP Benchmark

This repository contains the reproducibility artifacts for the HLRP benchmark
study. The release is intentionally compact: it keeps the common evaluator,
instance-generation scripts, baseline algorithms, compact MIP model, benchmark
instances, and table-level result summaries used in the manuscript.

## Layout

```text
data/
  main_core/          AP main-core JSON instances
  small_certified/    AP10/AP15 certified-grid JSON instances
  cross_source/       AP/CAB/TR scalability JSON instances

src/
  evaluator/          common feasibility and objective evaluator
  generator/          instance-generation scripts
  algorithms/         ALNS, GA, VNS, TS, and auxiliary SA baselines
  exact/              compact CPLEX MIP formulation

results/              curated CSV summaries used by the manuscript
```

Exploratory logs, plotting outputs, diagnostic probes, raw working folders, and
run-level JSON records are not included in this public release.

## Environment

Python 3.10 or later is recommended.

```bash
pip install -r requirements.txt
```

The heuristic baselines require only standard scientific Python packages. The
compact MIP model additionally requires IBM ILOG CPLEX and the Python packages
`cplex` and `docplex` in the active environment.

## Quick Checks

Evaluate one benchmark instance:

```bash
python src/evaluator/hlrp_instance.py --instance data/main_core/loose/ap10_loose_q5.json
```

Run one ALNS smoke test:

```bash
python src/algorithms/alns.py --instance_json data/main_core/loose/ap10_loose_q5.json --time_limit 5 --seed 11 --out scratch/ap10_loose_alns.json
```

Run the compact MIP when CPLEX is available:

```bash
python src/exact/compact_mip.py --instance_json data/main_core/loose/ap10_loose_q5.json --time_limit 600 --mip_gap 1e-9 --out_json scratch/ap10_loose_mip.json
```

## Result Summary

The `results/` directory contains one consolidated long-format table,
`summary_results.csv`, with four fields: `result_table`, `result_row`, `field`,
and `value`. The `result_table` field records the source summary represented by
each row group, so the numerical values reported in the manuscript can be
checked without retaining exploratory logs or run-level JSON records.
