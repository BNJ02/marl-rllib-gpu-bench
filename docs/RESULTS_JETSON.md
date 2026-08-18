# Résultats du sweep de convergence — Jetson Orin Nano

Sweep lancé le 2026-08-18/19. `MultiAgentPendulum` (2 agents), PPO, budget de
**300 000 pas d'environnement par run**, **3 graines par config**, CPU.
18 runs valides sur 24 tentés.

Machine : Jetson Orin Nano, 6 cœurs ARM, 8 Go de mémoire unifiée, Ray 2.57,
`OMP_NUM_THREADS=1`. Seuil de convergence : retour ≥ **-800** (sommé sur les
2 agents ; aléatoire ≈ -2400, plateau ≈ -300). Seuil fixé depuis la courbe du
run pilote, pas deviné.

## Ce qui est transférable à l'A40, et ce qui ne l'est pas

| Mesure | Dépend du hardware ? |
|---|---|
| Pas → seuil, AUC, verdict pur/non-pur | **Non** — arithmétique de gradient |
| Temps → seuil, débit | **Oui** |

Les colonnes « échantillons » ci-dessous valent donc telles quelles sur l'A40.
Les colonnes « temps » sont à rejouer.

## Résultats

| # | Levier | Pas → seuil | Temps → seuil | Débit | AUC (±σ) | Verdict |
|---|---|---:|---:|---:|---:|---|
| S1 | `batch=1024` | **48 469** (0,46×) | **155 s** (0,67×) | 314 | -606 ± 140 | **gain net** |
| S2 | baseline (4096/512) | 106 496 | 232 s | 465 | -917 ± 35 | référence |
| S3 | `batch=16384` | 294 912 (2,77×) | 582 s (2,51×) | 511 | -1936 ± 207 | compromis perdant |
| S4 | `batch=65536` | — | — | — | — | **non mesurable** (OOM) |
| S5 | `minibatch=128` | **60 075** (0,56×) | **189 s** (0,82×) | 320 | -612 ± 45 | **gain net** |
| S6 | `minibatch=4096` | jamais (3/3) | — | 513 | -2877 ± 251 | ne converge pas |
| S7 | `num_env_runners=1` | 107 861 (**1,013×**) | 549 s (2,36×) | 197 | -915 ± 63 | **pur** |
| S8 | `num_learners=1` | — | — | — | — | **non mesurable** (build) |

## Les trois questions de départ

### 1. Quels leviers sont purs ?

**`num_env_runners` est pur.** S7 (1 worker) consomme **1,3 % d'échantillons de
plus** que la baseline (4 workers) — à comparer à la variance inter-graines de
la baseline elle-même (4 %). Son AUC (-915 ± 63) est indiscernable de celle de
la baseline (-917 ± 35) : les courbes retour-vs-échantillons se superposent.
Le passage de 1 à 4 workers achète donc **2,36× de temps pour zéro coût en
apprentissage**. C'est le seul levier du sweep qu'on peut pousser sans
arbitrage.

### 2. Le gros batch dégrade-t-il la convergence ? Oui, massivement.

À ratio `batch/minibatch` constant (donc à nombre de gradient steps par
itération constant), l'efficacité-échantillon se dégrade de façon monotone :

| batch | pas → seuil | vs baseline |
|---|---:|---|
| 1 024 | 48 469 | 0,46× |
| 4 096 | 106 496 | 1× |
| 16 384 | 294 912 | 2,77× (+ 1 graine sur 3 censurée) |

Le mécanisme est direct : à budget d'échantillons fixe, un batch 16× plus gros
signifie 16× moins d'updates de policy. Le débit, lui, ne varie presque pas
(511 contre 465 éch/s, +10 %) — très loin de compenser un facteur 2,77.

**C'est la démonstration que le débit seul est une métrique trompeuse** : S3 est
la config la plus rapide du sweep en éch/s, et la pire en pratique. Un
classement au débit — celui du bench v1 — l'aurait sacrée gagnante.

### 3. Quand ça dégrade, y gagne-t-on au final ? Ici, jamais.

Aucune config ne réalise l'arbitrage espéré « je perds en échantillons mais je
gagne plus en temps ». S3 perd sur les deux axes (2,77× d'échantillons pour
2,51× de temps). Les deux configs gagnantes (S1, S5) gagnent sur les deux axes
à la fois. Sur cette tâche, **l'efficacité-échantillon domine le débit** —
l'échantillonnage n'est pas assez cher pour que l'arbitrage inverse existe.

## Le vrai paramètre de contrôle : les updates par échantillon

S1 et S5 atteignent le même nombre d'updates par échantillon par deux chemins
différents (petit batch / petit minibatch), et tous deux battent la baseline :

| config | updates / 1000 éch. | pas → seuil |
|---|---:|---:|
| S1 (batch 1024, mb 128) | 46,9 | 48 469 |
| S5 (batch 4096, mb 128) | 46,9 | 60 075 |
| S2 (batch 4096, mb 512) | 11,7 | 106 496 |
| S6 (batch 4096, mb 4096) | 1,5 | jamais |

Ce n'est donc ni la taille de batch ni celle du minibatch qui commande, mais
**leur rapport** — le nombre de fois où chaque échantillon sert à un gradient.
S6, à 1,5 update pour 1000 échantillons, n'apprend pas du tout (3 graines sur 3
au niveau aléatoire) : ce n'est pas « plus lent », c'est cassé.

Entre S1 et S5, à updates/échantillon égaux, S1 garde un léger avantage (48k
contre 60k), probablement parce qu'un batch 4× plus petit rafraîchit la policy
4× plus souvent, donc les updates portent sur des données moins périmées.

## Recommandation

**Baisser `minibatch_size` (S5) plutôt que le batch (S1).** S1 est un peu
meilleur en moyenne, mais bien moins fiable : σ de l'AUC à 140 contre 45 pour
S5, et une graine de S1 a mis 2,3× plus longtemps que les deux autres. S5 offre
presque le même gain avec une dispersion trois fois moindre.

Prédiction testable sur A40 : le seul coût de S5 est le surcoût CPU des updates
supplémentaires (débit 320 contre 465 éch/s). C'est précisément ce qu'un GPU
absorbe bien — **S5 devrait donc gagner encore plus largement sur A40**.

## Ce que le Jetson n'a pas pu mesurer

- **S4 (`batch=65536`)** : OOM des EnvRunners sur les 3 graines, 0 échantillon
  collecté. Retenté avec 2 workers (`S4b`) : échoue également. La mémoire
  unifiée de 8 Go est saturée par le volume d'épisodes, indépendamment du
  découpage. **À mesurer sur A40.**
- **S8 (`num_learners=1`)** : `torch.distributed.is_available() == False` — le
  wheel torch NVIDIA JetPack est compilé sans, or RLlib l'exige dès
  `num_learners ≥ 1`. Rien à voir avec la convergence, le run n'a jamais
  démarré. **Mesurable sur A40** (torch standard).
- **Le GPU du Jetson** : écarté sur mesure, pas par renoncement.
  `ray.init()` consomme ~2,9 Go de mémoire unifiée (5,20 → 2,26 Go de GPU
  libre), il ne reste pas de quoi allouer le contexte cuBLAS du learner —
  `CUBLAS_STATUS_ALLOC_FAILED` même à 1 worker, alors qu'un `torch` nu alloue
  et fait un forward sans problème. Sans conséquence sur les conclusions
  ci-dessus : le v1 avait déjà mesuré que ce GPU n'apporte que 0-7 % sur cette
  charge, et la moitié « échantillons » ne dépend ni du device ni de la machine.

## Limites

- 3 graines. Le bruit inter-graines n'est **pas uniforme** : 4 % sur la
  baseline, 130 % sur S1. Aucune barre d'erreur globale n'a donc de sens, d'où
  les bandes min/max par config dans le notebook. Les écarts inférieurs à ~2×
  sur les configs à petit batch ne sont pas concluants.
- Le premier franchissement de seuil est bruité par construction ; l'AUC (qui
  intègre toute la courbe) est la métrique de classement primaire.
- Une tâche (`MultiAgentPendulum`, 2 agents), un réseau (MLP par défaut). Le
  point d'équilibre entre efficacité-échantillon et débit dépend du coût de
  simulation de l'env : sur un env beaucoup plus cher à simuler, l'arbitrage
  pourrait s'inverser.
