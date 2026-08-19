"""Tableau récapitulatif de la matrice de débit.

    python -m bench.summarize_matrix [run_id]

Le débit est calculé hors 1re itération (warm-up JIT/graph), et rapporté au
**batch effectif** (`train_batch_size_per_learner` x nb de learners) : sans ça
une config à num_learners=2 paraîtrait deux fois plus rapide alors qu'elle
traite simplement deux fois plus de données par itération.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

LOGS = Path(__file__).resolve().parent.parent / "logs"


def main() -> None:
    run_id = sys.argv[1] if len(sys.argv) > 1 else "*"
    rows_by_cfg: dict[str, list[dict]] = {}
    for f in sorted(glob.glob(str(LOGS / f"matrix_{run_id}_*.jsonl"))):
        rows = [json.loads(l) for l in open(f) if l.strip()]
        if rows:
            rows_by_cfg[rows[-1]["bench_config/name"]] = rows

    if not rows_by_cfg:
        print("aucun log matrix trouvé")
        return

    base = None
    results = []
    for name, rows in sorted(rows_by_cfg.items()):
        steady = [r for r in rows if not r["is_warmup"]]
        if not steady:
            continue
        last, first = rows[-1], steady[0]
        # temps du régime établi = total - temps de la 1re itération
        t_steady = last["cumulative_time_s"] - rows[0]["wall_time_s"]
        n_steps = last["env_steps_sampled"] - rows[0]["env_steps_sampled"]
        sps = n_steps / t_steady if t_steady > 0 else 0
        r = {
            "name": name,
            "lever": last["bench_config/lever"],
            "sps": sps,
            "s_per_iter": t_steady / len(steady),
            "eff_batch": last["bench_config/effective_batch"],
            "note": last.get("bench_config/note", ""),
        }
        results.append(r)
        # Référence = la config "00" de la matrice utilisée (M00 pour MATRIX,
        # A00 pour MATRIX_A40). Chercher "M00" en dur laissait la colonne
        # "vs réf" vide sur toute matrice ne commençant pas par M.
        if name.endswith("00"):
            base = r

    print(f"{'cfg':5s} {'levier':38s} {'éch/s':>9s} {'s/iter':>7s} "
          f"{'batch eff':>10s}  {'vs réf':>7s}")
    print("-" * 88)
    for r in results:
        rel = f"{r['sps']/base['sps']:.2f}x" if base and base["sps"] else "-"
        star = " *" if base and r["sps"] > base["sps"] else ""
        print(f"{r['name']:5s} {r['lever']:38s} {r['sps']:9.0f} {r['s_per_iter']:7.2f} "
              f"{r['eff_batch']:10d}  {rel:>7s}{star}")
        if r["note"]:
            print(f"      ^ {r['note']}")


if __name__ == "__main__":
    main()
