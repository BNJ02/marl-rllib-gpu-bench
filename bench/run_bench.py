"""CLI : lance un ou plusieurs des 5 configs A-E, log exhaustif en JSONL.

    uv run python bench/run_bench.py --configs A,B,C,D,E --iters 5
    uv run python bench/run_bench.py --configs A --iters 2 --force-cpu   # smoke-test sans GPU

Un fichier `logs/<run_id>_<config>.jsonl` par config. `notebooks/
analyze_results.ipynb` les relit tous pour les graphes.
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")  # AVANT tout import torch/ray
# Lancé avec `uv run`, Ray relance sinon ses EnvRunners via `uv run` eux aussi
# (chacun recrée un venv complet, lent et inutile ici — le venv du driver
# suffit). Même fix que simple_rl/train.py. À poser avant tout import ray.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

import torch  # noqa: E402

from bench.configs import CONFIGS, build_config  # noqa: E402
from bench.logging_utils import JsonlLogger, run_metadata  # noqa: E402

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", default="A,B,C,D,E", help="sous-ensemble, ex. A,C")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--cpu-count", type=int, default=os.cpu_count(),
                         help="cœurs disponibles (résout num_env_runners = cœurs-2/3)")
    parser.add_argument("--force-cpu", action="store_true",
                         help="ignore le GPU même pour B-E (smoke-test sans carte)")
    parser.add_argument(
        "--jetson-workaround", action="store_true",
        help="desactive cuDNN (bug connu du wheel torch NVIDIA pré-release JetPack, "
             "n'a rien à faire sur un GPU standard type A40)",
    )
    args = parser.parse_args()

    if args.jetson_workaround:
        torch.backends.cudnn.enabled = False

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    meta = run_metadata(args.cpu_count)
    print(f"run_id={run_id}  {meta}", flush=True)

    names = args.configs.split(",")
    for name in names:
        spec = CONFIGS[name]
        cfg = build_config(spec, args.cpu_count, force_cpu=args.force_cpu)
        workers = spec.resolve_workers(args.cpu_count)
        use_gpu = spec.use_gpu and not args.force_cpu

        print(f"\n=== config {name} ({spec.label}) : batch={spec.batch} "
              f"workers={workers} gpu={use_gpu} learners={spec.num_learners} ===", flush=True)

        algo = cfg.build_algo()
        log_path = LOGS_DIR / f"{run_id}_{name}.jsonl"
        logger = JsonlLogger(log_path, {"run_id": run_id, **meta})

        config_spec_row = {
            "name": spec.name, "label": spec.label, "batch": spec.batch,
            "minibatch": spec.minibatch, "workers": workers, "use_gpu": use_gpu,
            "num_learners": spec.num_learners, "hiddens": spec.hiddens,
        }

        t_cumulative = 0.0
        for i in range(args.iters):
            t0 = time.perf_counter()
            result = algo.train()
            dt = time.perf_counter() - t0
            t_cumulative += dt
            logger.log_iteration(
                iteration=i, is_warmup=(i == 0), wall_time_s=dt,
                cumulative_time_s=t_cumulative, config_spec=config_spec_row, result=result,
            )
            print(f"  iter {i} : {dt:.2f}s  throughput={spec.batch/dt:.0f} ech/s", flush=True)

        logger.close()
        algo.stop()
        print(f"  -> {log_path}", flush=True)

    print(f"\nlogs écrits dans {LOGS_DIR}/{run_id}_*.jsonl", flush=True)


if __name__ == "__main__":
    main()
