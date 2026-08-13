# marl-rllib-gpu-bench

Cinq configs RLlib PPO (multi-agent, `MultiAgentCartPole`), CPU vs GPU, pour
répondre à une question précise : **une fois qu'on a un vrai GPU (A40) et 8
cœurs, quels réglages RLlib en tirent vraiment parti, et lesquels ne changent
rien ?**

Suite de [`SOTA_MARL_RLlib`](https://github.com/BNJ02/marl-rllib-sota)
(état de l'art MARL) et de son bench sur Jetson Orin Nano — où la conclusion
avait été : sur un petit réseau/batch, le GPU n'apporte quasiment rien, le
vrai levier est `num_env_runners`. Reste à vérifier si ça tient sur du vrai
matériel de calcul (A40) ou si c'était un artefact du Jetson (GPU embarqué,
mémoire unifiée, torch pré-release).

## Les 5 configs (`bench/configs.py`)

| # | Ce qu'on isole | batch | minibatch | GPU | learners | réseau |
|---|---|---:|---:|---|---|---|
| A | référence CPU | 2 048 | 256 | non | 0 (local) | défaut |
| B | GPU à petit batch | 2 048 | 256 | oui | 0 (local) | défaut |
| C | GPU à gros batch | 32 768 | 4 096 | oui | 0 (local) | défaut |
| D | GPU + réseau plus lourd | 32 768 | 4 096 | oui | 0 (local) | `[512, 512]` |
| E | learner en acteur Ray dédié | 32 768 | 4 096 | oui | 1 | défaut |

`num_env_runners` = cœurs disponibles − 2 (− 3 pour E, qui réserve un cœur de
plus au process driver). Overridable via `--cpu-count`.

- **A vs B** : le GPU seul, à batch inchangé, rattrape-t-il le CPU ?
- **B vs C** : 16× le batch, même nombre de gradient steps (minibatch
  proportionnel) — c'est l'axe où le GPU est censé décrocher.
- **C vs D** : si C ne suffit pas, est-ce la taille du **réseau** qui manquait ?
- **C vs E** : sortir le learner du process principal (acteur Ray dédié) —
  coûte un cœur, se généralise au multi-GPU.

Détail complet, incluant le "pourquoi" de chaque choix (thread-thrashing
OpenMP, coût des policies séquentielles en MARL, etc.) : issu de
`SOTA_MARL_RLlib/docs/07-scaling-et-perf.md` et du bench Jetson associé.

## Installation

Deux façons, mêmes dépendances (`pyproject.toml`) :

```bash
# uv (recommandé en dev)
uv sync

# pip + venv classique (recommandé sur la VM A40 — proxy pip interne déjà
# validé pour ce type d'install, cf. remarque plus bas)
python3 -m venv .venv
.venv/bin/pip install -e .
```

## Lancer le bench

```bash
uv run python -m bench.run_bench --configs A,B,C,D,E --iters 5
# ou un sous-ensemble :
uv run python -m bench.run_bench --configs A,C --iters 5
# smoke-test sans GPU (ignore --use-gpu de B-E, ne change rien d'autre) :
uv run python -m bench.run_bench --configs A --iters 2 --force-cpu
```

Un fichier `logs/<run_id>_<config>.jsonl` par config, log exhaustif (voir
[`logs/README.md`](logs/README.md) pour le schéma). Rien n'est écrasé — les
runs s'accumulent.

## Analyser les résultats

```bash
uv run jupyter lab notebooks/analyze_results.ipynb
```

Charge tous les `logs/*.jsonl`, produit débit par config (avec barres
d'erreur), trajectoire temps/itération (warmup visible), mémoire GPU si
disponible, et une comparaison avec les chiffres déjà mesurés sur Jetson.

## Déployer sur la VM A40

D'après [`~/RL/docs/CLUSTER_A40_TEST.md`](../RL/docs/CLUSTER_A40_TEST.md)
(pas public, notes internes) : la VM A40 a ses 8 cœurs CPU **et** le GPU sur
la même machine — pas besoin de cluster Ray multi-nœud pour ce bench, tout
tourne en local (`num_env_runners` + `num_gpus_per_learner` suffisent, comme
pour n'importe quel poste avec GPU).

```bash
# depuis la VM A40
git clone https://github.com/BNJ02/marl-rllib-gpu-bench
cd marl-rllib-gpu-bench
python3 -m venv .venv && .venv/bin/pip install -e .
.venv/bin/python -m bench.run_bench --configs A,B,C,D,E --iters 5
```

Torch standard (pas de wheel pré-release type JetPack) : aucun patch runtime
requis. Si un jour ce bench tourne sur un Jetson (GPU embarqué, wheel
NVIDIA pré-release), passer `--jetson-workaround` (désactive cuDNN — bug
connu de segfault au premier forward GPU sur ce type de build).

Récupérer les logs vers une autre machine pour l'analyse :

```bash
rsync -av a40:~/marl-rllib-gpu-bench/logs/*.jsonl ./logs/
```

## Ce que ce repo ne fait pas

- Pas de cluster Ray multi-nœud (c'est un sujet à part, cf. les notes
  internes `~/RL` sur le montage tête Windows + worker A40 par SSH/TLS —
  hors scope ici, ce bench est mono-machine).
- Pas d'entraînement "réel" jusqu'à convergence : 5 itérations par config,
  assez pour mesurer un débit stable, pas pour juger une politique.
