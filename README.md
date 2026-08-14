# marl-rllib-gpu-bench

Deux bancs d'essai complémentaires sur RLlib PPO multi-agent :

- **v1 — débit** ([§ Bench de débit](#bench-de-débit-v1)) : 5 configs A-E,
  combien d'échantillons par seconde, CPU vs GPU.
- **v2 — convergence** ([§ Bench de convergence](#bench-de-convergence-v2)) :
  8 configs S1-S8, est-ce que le débit gagné coûte de la vitesse
  d'apprentissage ? Un réglage « 2× plus rapide » qui a besoin de 3× plus
  d'échantillons est une régression déguisée — le v1 seul ne peut pas le voir.

---

## Bench de débit (v1)

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

---

## Bench de convergence (v2)

Le débit seul est une métrique trompeuse : augmenter le batch améliore les
échantillons/seconde, mais réduit le nombre de gradient steps par échantillon.
La vraie question n'est pas « combien d'échantillons par seconde » mais
**« combien de temps pour atteindre un niveau de performance donné »**.

### La moitié des résultats ne dépend pas de la machine

C'est la propriété qui structure tout ce bench :

| Ce qu'on mesure | Dépend du hardware ? |
|---|---|
| Courbe retour vs **échantillons consommés** | **Non** — arithmétique de gradient pure |
| Verdict pur / non-pur d'un levier | **Non** (déduit de la courbe ci-dessus) |
| `samples_to_threshold` (efficacité-échantillon) | **Non** |
| Débit, `time_to_threshold` | **Oui** |
| Verdict final « quelle config gagne » | **Oui** (c'est le ratio des deux) |

Conséquence pratique : un sweep lancé sur une petite machine (Jetson) tranche
**définitivement** la question « le gros batch dégrade-t-il la convergence, et
de combien ». Seule la moitié chronométrée est à rejouer sur l'A40.

### Les 8 configs (`bench/sweep.py`)

Env : `MultiAgentPendulum` (2 agents) — CartPole sature en ~20 itérations, donc
à gros batch il ne resterait que 2-3 itérations, aucune résolution pour mesurer
une dégradation de convergence. Baseline S2 : `batch=4096, minibatch=512,
epochs=6, workers=4, GPU`. Un seul facteur change à la fois.

| # | Levier isolé | Hypothèse |
|---|---|---|
| S1 | `batch=1024` (mb 128) | non pur : meilleure eff.-éch., débit moindre |
| S2 | **baseline** | référence |
| S3 | `batch=16384` (mb 2048) | non pur : dégradation attendue |
| S4 | `batch=65536` (mb 8192) | non pur : dégradation forte |
| S5 | `minibatch=128` | non pur : 8× plus de gradient steps |
| S6 | `minibatch=4096` | non pur : 1 seul step par epoch |
| S7 | `num_env_runners=1` | **pur attendu** |
| S8 | `num_learners=1` (acteur dédié) | **pur attendu** |

S1-S4 gardent le ratio `batch/minibatch = 8` constant, donc le même nombre de
gradient steps **par itération** (48) — mais à budget d'échantillons fixe, le
nombre d'updates **par échantillon** varie d'un facteur 64 (46.9 pour 1000 pas
en S1, 0.73 en S4). C'est précisément le mécanisme par lequel le gros batch
peut dégrader la convergence. S5-S6 font l'inverse (batch fixe, ratio variable)
pour démêler « taille de batch » de « nombre d'updates ».

Contrôle utile qui tombe du design : **S1 et S5 font le même nombre d'updates
par échantillon (46.9/1k) avec des batchs 4× différents** — leur écart isole
donc l'effet du *bruit du gradient* seul.

Deux pièges encodés dans le code :
- `train_batch_size_per_learner` est **par learner** : `num_learners=2`
  doublerait silencieusement le batch effectif. S8 reste donc à
  `num_learners=1` (placement, pas parallélisme) pour rester un test pur.
- `episode_return_mean` est **sommé sur les agents** en multi-agent : sur
  Pendulum 2 agents, aléatoire ≈ -2400, bon ≈ -400.

### Leviers volontairement non testés

`num_cpus_per_env_runner`, `num_gpus_per_env_runner`, `num_cpus_per_learner`,
`evaluation_num_env_runners` : purs **par construction** (allocation de
ressources, maths d'optimisation identiques). Leur seul effet possible est le
chrono, déjà couvert par le v1 — les inclure dans un sweep de convergence
brûlerait du budget pour confirmer une tautologie.
`num_envs_per_env_runner` : les envs multi-agents ne sont pas vectorisables
dans RLlib, gain attendu nul.

### Lancer le sweep

```bash
# pilote d'abord : vérifier que Pendulum apprend, mesurer le débit réel,
# et FIXER le seuil depuis la courbe observée (ne pas le deviner)
python -m bench.run_sweep --configs S2 --seeds 0 --max-env-steps 300000

# sweep complet (~4h sur Jetson) ; --resume rend l'interruption sans danger
python -m bench.run_sweep --configs S1,S2,S3,S4,S5,S6,S7,S8 \
    --seeds 0,1,2 --max-env-steps 300000 --resume --run-id monsweep
```

Budget en **pas d'environnement**, pas en itérations : à budget d'itérations
égal, une config à batch 65536 recevrait 16× plus de données et la comparaison
ne voudrait rien dire. Un OOM sur une config (fréquent à gros batch sur les 8
Go partagés du Jetson) est capturé et n'interrompt pas les autres runs.

Analyse : `notebooks/analyze_convergence.ipynb` — courbes retour-vs-échantillons
(test de pureté), retour-vs-temps, table `samples/time_to_threshold` avec les
runs censurés conservés, et le nuage « ce qu'on perd en échantillons vs ce
qu'on gagne en temps ». **Ajuster `THRESHOLD` en tête de notebook** d'après le
pilote.

---

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
