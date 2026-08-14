# Format des logs

Deux familles de fichiers, une ligne JSON par itération dans les deux cas :

- `<run_id>_<config>.jsonl` — bench de **débit** (v1), `config` ∈ `{A..E}`
  (cf. `bench/configs.py`), lu par `notebooks/analyze_results.ipynb`.
- `sweep_<run_id>_<config>_seed<N>.jsonl` — bench de **convergence** (v2),
  `config` ∈ `{S1..S8}` (cf. `bench/sweep.py`), lu par
  `notebooks/analyze_convergence.ipynb`. Ajoute `seed` et `env_steps_sampled`
  (l'axe x « échantillons », celui dont les conclusions ne dépendent pas du
  hardware).

`run_id` = timestamp UTC (`YYYYMMDDThhmmssZ`) sauf si passé via `--run-id`.

Chaque ligne contient :
- métadonnées de run : `hostname`, `platform`, `cpu_count_detected`,
  `gpu_name`, `torch_version`, `cuda_available`, `git_commit`.
- `iter`, `is_warmup` (la 1ʳᵉ itération, gardée — pas moyennée en silence,
  elle inclut le JIT/graph build, visible dans le notebook).
- `wall_time_s`, `cumulative_time_s`, `throughput_sps` (= `batch / wall_time_s`).
- `bench_config/*` : les 8 champs de la config lancée (name, label, batch,
  minibatch, workers, use_gpu, num_learners, hiddens).
- `gpu_mem_allocated_mb` / `gpu_mem_reserved_mb` (`None` si pas de GPU).
- **tout** le dict de résultats RLlib aplati (`env_runners/...`,
  `learners/shared/...`, `config/...` — ceci est le dump interne de RLlib,
  distinct de `bench_config/*`) : ~300 clés, rien n'est trié à la main.

## `sweep_SMOKE_S{1,2}_seed{0,1}.jsonl`

**Smoke-test du pipeline de convergence, pas des résultats.** 4000 pas par run
sur une machine de dev CPU : les retours restent à ~-2400 (niveau aléatoire),
aucun run n'atteint le seuil. Sert uniquement à prouver que
`run_sweep.py` → JSONL → `analyze_convergence.ipynb` fonctionne, y compris le
traitement des runs **censurés** (jamais convergés). À supprimer ou ignorer dès
qu'un vrai sweep existe.

## `example_smoke_test_local_cpu.jsonl`

**Données réelles, mais CPU seul, sur une machine de dev sans GPU — pas
l'A40.** Sert juste à vérifier que le pipeline logs → notebook fonctionne
avant d'avoir un vrai run. À remplacer (ou compléter) par un run réel :

```bash
python -m bench.run_bench --configs A,B,C,D,E --iters 5
```

Le notebook lit tous les `*.jsonl` de ce dossier — les nouveaux runs
s'ajoutent, rien n'est écrasé.
