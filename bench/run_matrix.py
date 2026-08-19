"""Lance la matrice de débit : N itérations par config, un JSONL par config.

    python -m bench.run_matrix --configs M00,M01 --iters 5
    python -m bench.run_matrix --iters 5            # toute la matrice

Ne mesure QUE le débit (la moitié dépendante du matériel). La convergence est
mesurée séparément par `run_sweep.py`, une seule fois, sur n'importe quelle
machine — voir l'en-tête de `bench/matrix.py`.

Chaque config tourne dans un cluster Ray neuf (`ray.init`/`ray.shutdown` par
config) : sans ça les acteurs d'une config fuient sur la suivante et faussent
les mesures des configs à beaucoup de workers.
"""

from __future__ import annotations

import argparse
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

import ray  # noqa: E402

from bench.logging_utils import JsonlLogger, run_metadata  # noqa: E402
from bench import matrix as matrix_mod  # noqa: E402
from bench.matrix import MATRIX, build_matrix_config  # noqa: E402

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
MAX_STALLED_ITERS = 3


def run_one(spec, args, run_id: str, meta: dict) -> str:
    log_path = LOGS_DIR / f"matrix_{run_id}_{spec.name}.jsonl"
    if args.resume and log_path.exists():
        with open(log_path) as fh:
            if sum(1 for l in fh if l.strip()) >= args.iters:
                print(f"  [skip] {spec.name} déjà mesuré", flush=True)
                return "skipped"

    cfg = build_matrix_config(spec, seed=0)
    algo = cfg.build_algo()
    logger = JsonlLogger(log_path, {"run_id": run_id, "seed": 0, **meta})

    row = {
        "name": spec.name, "lever": spec.lever, "batch": spec.batch,
        "minibatch": spec.minibatch, "effective_batch": spec.effective_batch,
        "workers": spec.workers, "envs_per_runner": spec.envs_per_runner,
        "cpus_per_runner": spec.cpus_per_runner, "gpus_per_runner": spec.gpus_per_runner,
        "num_learners": spec.num_learners, "cpus_per_learner": spec.cpus_per_learner,
        "gpus_per_learner": spec.gpus_per_learner, "eval_workers": spec.eval_workers,
        "hiddens": str(spec.hiddens), "note": spec.note,
    }

    steps, t_cum, stalled = 0, 0.0, 0
    for i in range(args.iters):
        t0 = time.perf_counter()
        result = algo.train()
        dt = time.perf_counter() - t0
        t_cum += dt
        reported = int(result.get("env_runners", {}).get("num_env_steps_sampled_lifetime", 0))
        if reported <= steps:
            stalled += 1
            if stalled >= MAX_STALLED_ITERS:
                raise RuntimeError(f"aucun échantillon depuis {stalled} itérations")
        else:
            stalled = 0
        steps = max(steps, reported)
        logger.log_iteration(
            iteration=i, is_warmup=(i == 0), wall_time_s=dt, cumulative_time_s=t_cum,
            config_spec=row, result=result, extra={"env_steps_sampled": steps},
        )
        print(f"  it {i} : {dt:6.2f}s  {spec.effective_batch/dt:8.0f} éch/s", flush=True)

    logger.close()
    algo.stop()
    # Débit régime établi : on exclut la 1re itération (warm-up JIT/graph).
    print(f"  -> TERMINÉ {spec.name} : {steps} pas en {t_cum:.1f}s", flush=True)
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", default=None)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--ray-cpus", type=int, default=None,
                     help="limite les CPU vus par Ray (émulation d'une machine plus petite ; "
                          "sans ça son ordonnanceur croit disposer de toute la machine)")
    ap.add_argument("--matrix", default="MATRIX", choices=["MATRIX", "MATRIX_A40"],
                     help="quel dict de configs utiliser")
    args = ap.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    meta = run_metadata(os.cpu_count())
    print(f"run_id={run_id}  {meta}\n", flush=True)

    table = getattr(matrix_mod, args.matrix)
    names = args.configs.split(",") if args.configs else list(table)
    outcomes: dict[str, str] = {}
    for name in names:
        spec = table[name]
        print(f"=== {spec.name} : {spec.lever} ===", flush=True)
        try:
            # Cluster neuf par config : sinon les acteurs de la config
            # précédente restent alloués et faussent les mesures suivantes.
            ray.init(include_dashboard=False, log_to_driver=False,
                      **({"num_cpus": args.ray_cpus} if args.ray_cpus else {}))
            outcomes[spec.name] = run_one(spec, args, run_id, meta)
        except Exception as exc:
            outcomes[spec.name] = f"FAILED: {type(exc).__name__}"
            print(f"  !! {spec.name} a échoué : {exc}", flush=True)
            traceback.print_exc()
        finally:
            ray.shutdown()

    print("\n=== RÉCAPITULATIF ===", flush=True)
    for k, v in outcomes.items():
        print(f"{k:6s} {v}", flush=True)


if __name__ == "__main__":
    main()
