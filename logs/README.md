# Format des logs

Un fichier `<run_id>_<config>.jsonl` par config lancée, une ligne JSON par
itération. `run_id` = timestamp UTC (`YYYYMMDDThhmmssZ`), `config` ∈
`{A,B,C,D,E}` (cf. `bench/configs.py`).

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

## `example_smoke_test_local_cpu.jsonl`

**Données réelles, mais CPU seul, sur une machine de dev sans GPU — pas
l'A40.** Sert juste à vérifier que le pipeline logs → notebook fonctionne
avant d'avoir un vrai run. À remplacer (ou compléter) par un run réel :

```bash
python -m bench.run_bench --configs A,B,C,D,E --iters 5
```

Le notebook lit tous les `*.jsonl` de ce dossier — les nouveaux runs
s'ajoutent, rien n'est écrasé.
