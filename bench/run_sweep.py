"""Sweep de convergence : configs x graines, budget en PAS D'ENVIRONNEMENT.

    python -m bench.run_sweep --configs S1,S2,S3,S4,S5,S6,S7,S8 \
        --seeds 0,1,2 --max-env-steps 300000

Budget en pas d'env, pas en itérations : à budget d'itérations égal, une config
à batch 65536 recevrait 16x plus de données qu'une config à batch 4096 et la
comparaison ne voudrait rien dire.

Un fichier `logs/sweep_<run_id>_<config>_seed<N>.jsonl` par (config, graine).
Reprise : un couple déjà loggé et terminé est skippé (indispensable sur un
sweep de plusieurs heures qui peut être interrompu).
"""

from __future__ import annotations

import argparse
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")  # AVANT tout import torch/ray
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

import torch  # noqa: E402

from bench.logging_utils import JsonlLogger, run_metadata  # noqa: E402
from bench.sweep import SWEEP, build_sweep_config  # noqa: E402

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
STEPS_KEY = "env_runners/num_env_steps_sampled_lifetime"


def _already_done(path: Path, max_env_steps: int) -> bool:
    """Un run est considéré terminé si son log existe et que sa dernière ligne
    a atteint le budget. Un run interrompu à mi-parcours est donc relancé."""
    if not path.exists():
        return False
    try:
        import json
        last = None
        with open(path) as fh:
            for line in fh:
                if line.strip():
                    last = json.loads(line)
        return last is not None and last.get("env_steps_sampled", 0) >= max_env_steps
    except Exception:
        return False


def run_one(spec, seed: int, args, run_id: str, meta: dict) -> str:
    log_path = LOGS_DIR / f"sweep_{run_id}_{spec.name}_seed{seed}.jsonl"
    if args.resume and _already_done(log_path, args.max_env_steps):
        print(f"  [skip] {spec.name} seed={seed} déjà terminé", flush=True)
        return "skipped"

    cfg = build_sweep_config(spec, seed, force_cpu=args.force_cpu)
    algo = cfg.build_algo()
    logger = JsonlLogger(log_path, {"run_id": run_id, "seed": seed, **meta})

    spec_row = {
        "name": spec.name, "lever": spec.lever, "hypothesis": spec.hypothesis,
        "batch": spec.batch, "minibatch": spec.minibatch, "workers": spec.workers,
        "num_learners": spec.num_learners,
        "use_gpu": spec.use_gpu and not args.force_cpu,
        "seed": seed,
    }

    steps, iteration, t_cumulative = 0, 0, 0.0
    while steps < args.max_env_steps:
        t0 = time.perf_counter()
        result = algo.train()
        dt = time.perf_counter() - t0
        t_cumulative += dt
        steps = int(result.get("env_runners", {}).get("num_env_steps_sampled_lifetime", 0))

        logger.log_iteration(
            iteration=iteration, is_warmup=(iteration == 0), wall_time_s=dt,
            cumulative_time_s=t_cumulative, config_spec=spec_row, result=result,
            extra={"env_steps_sampled": steps},
        )
        ret = result.get("env_runners", {}).get("episode_return_mean", float("nan"))
        print(f"  it {iteration:>3} | steps {steps:>7} | return {ret:9.1f} | {dt:6.2f}s",
              flush=True)
        iteration += 1

    logger.close()
    algo.stop()
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", default=",".join(SWEEP))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--max-env-steps", type=int, default=300_000)
    ap.add_argument("--force-cpu", action="store_true")
    ap.add_argument("--resume", action="store_true",
                     help="skippe les (config, graine) déjà terminés")
    ap.add_argument("--run-id", default=None,
                     help="réutiliser un run_id existant (pour --resume)")
    ap.add_argument("--jetson-workaround", action="store_true",
                     help="désactive cuDNN (wheel torch NVIDIA pré-release JetPack)")
    args = ap.parse_args()

    if args.jetson_workaround:
        torch.backends.cudnn.enabled = False

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    meta = run_metadata(os.cpu_count())
    print(f"run_id={run_id}  budget={args.max_env_steps} pas/run  {meta}\n", flush=True)

    names = args.configs.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    outcomes: dict[str, str] = {}

    for name in names:
        spec = SWEEP[name]
        for seed in seeds:
            tag = f"{name}/seed{seed}"
            print(f"=== {tag} : {spec.lever} (batch={spec.batch} mb={spec.minibatch} "
                  f"workers={spec.workers} learners={spec.num_learners}) ===", flush=True)
            try:
                outcomes[tag] = run_one(spec, seed, args, run_id, meta)
            except Exception as exc:
                # Un OOM (fréquent sur Jetson à gros batch, mémoire unifiée 8 Go)
                # ne doit pas tuer les 23 autres runs du sweep.
                outcomes[tag] = f"FAILED: {type(exc).__name__}"
                print(f"  !! {tag} a échoué : {exc}", flush=True)
                traceback.print_exc()

    print("\n=== RÉCAPITULATIF ===", flush=True)
    for tag, outcome in outcomes.items():
        print(f"{tag:20s} {outcome}", flush=True)
    print(f"\nlogs : {LOGS_DIR}/sweep_{run_id}_*.jsonl", flush=True)


if __name__ == "__main__":
    main()
