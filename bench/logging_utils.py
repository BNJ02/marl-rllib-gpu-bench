"""Aplatit les résultats RLlib et écrit un log JSONL, une ligne par itération.

Rien n'est trié à la main : tout le dict de résultats RLlib est aplati et
loggé (même pattern que `simple_rl/train.py::_flatten`), pour ne perdre
aucune métrique produite par RLlib (retour, entropie, vf_explained_var, KL,
pertes par policy...).
"""

from __future__ import annotations

import json
import numbers
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch


def flatten(obj: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from flatten(value, f"{prefix}/{key}" if prefix else str(key))
    elif isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            yield from flatten(value, f"{prefix}/{idx}")
    else:
        yield prefix, obj


def numeric_metrics(result: dict) -> dict[str, float]:
    """Toutes les feuilles numériques finies du résultat RLlib, aplaties."""
    return {
        k: float(v)
        for k, v in flatten(result)
        if isinstance(v, numbers.Number) and np.isfinite(v)
    }


def gpu_memory_mb() -> dict[str, float | None]:
    if not torch.cuda.is_available():
        return {"gpu_mem_allocated_mb": None, "gpu_mem_reserved_mb": None}
    return {
        "gpu_mem_allocated_mb": torch.cuda.memory_allocated() / 1e6,
        "gpu_mem_reserved_mb": torch.cuda.memory_reserved() / 1e6,
    }


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def run_metadata(cpu_count: int) -> dict:
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "cpu_count_detected": cpu_count,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "git_commit": git_commit(),
    }


class JsonlLogger:
    """Un fichier JSONL par config. `log_iteration` écrit une ligne (métadonnées
    d'itération + métriques RLlib aplaties) et flush immédiatement — un run
    long qui plante en cours de route ne doit pas perdre les lignes déjà
    écrites."""

    def __init__(self, path: Path, run_meta: dict):
        self.path = path
        self.run_meta = run_meta
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w")

    def log_iteration(self, *, iteration: int, is_warmup: bool, wall_time_s: float,
                       cumulative_time_s: float, config_spec: dict, result: dict,
                       extra: dict | None = None) -> None:
        row = {
            **self.run_meta,
            "iter": iteration,
            "is_warmup": is_warmup,
            "wall_time_s": wall_time_s,
            "cumulative_time_s": cumulative_time_s,
            "throughput_sps": config_spec["batch"] / wall_time_s,
            **(extra or {}),
            # préfixe distinct de "config/*" : RLlib aplatit son PROPRE dump de
            # config dans `result` (des centaines de clés internes), on ne
            # veut pas que nos 8 champs s'y mélangent.
            **{f"bench_config/{k}": v for k, v in config_spec.items()},
            **gpu_memory_mb(),
            **numeric_metrics(result),
        }
        self._f.write(json.dumps(row) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def config_spec_dict(spec) -> dict:
    d = asdict(spec)
    d.pop("workers_offset", None)
    return d
