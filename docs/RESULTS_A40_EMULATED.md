# Machine cible émulée : 8 cœurs + 1 GPU type A40

Les recommandations pour une VM « 8 cœurs + A40 » étaient **extrapolées** de
deux régimes encadrants — Jetson (CPU-pauvre, GPU-pauvre) et cluster dataia25
non contraint (CPU-riche, GPU-riche). La cible est dans un troisième régime,
CPU-pauvre et GPU-riche, où aucune des deux mesures ne s'applique.
Cette campagne le mesure directement, et **corrige trois recommandations
erronées**.

## Comment la cible est émulée

| Contrainte | Moyen | Preuve dans chaque JSONL |
|---|---|---|
| 8 cœurs physiques | `taskset -c 0-7` | `cpu_affinity: 8` |
| 1 GPU type A40 | `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0` | `gpu_name: RTX 3090`, `gpu_count_visible: 1` |
| Ray limité à 8 CPU | `--ray-cpus 8` | — |

Les CPU logiques 0-63 de l'EPYC 7B13 sont des cœurs physiques distincts (les
siblings SMT sont en 64-127, vérifié via `thread_siblings_list`) : `taskset -c
0-7` donne donc 8 vrais cœurs, sans partage SMT.

**La 3090 comme proxy d'A40** : même silicium GA102 Ampere, 10 496 contre
10 752 cœurs CUDA, 35,6 contre 37,4 TFLOPS fp32 — moins de 5 % d'écart. La 4090
(AD102, ~82 TFLOPS) aurait été un mauvais proxy, et c'est pourtant elle que
torch place en device 0 sans `CUDA_DEVICE_ORDER=PCI_BUS_ID`.

**Piège corrigé** : `os.cpu_count()` renvoie 128 **même sous `taskset`**.
`run_metadata` logge désormais `os.sched_getaffinity(0)` : sans ça, toute la
campagne aurait certifié une contrainte non appliquée.

**Limite** : c'est une émulation. Les cœurs EPYC 7B13, la bande passante
mémoire et la topologie PCIe ne sont pas ceux de la VM cible. Ça borne la
réponse, ça ne remplace pas un run réel.

---

## Matrice de débit sous contrainte (13 configs)

| cfg | levier | éch/s | vs réf |
|---|---|---:|---:|
| A09 | `batch=16384` | 3 598 | 3,13× |
| A11 | `batch=4096, mb=2048` | 3 550 | 3,09× |
| A08 | `batch=4096` | 2 388 | 2,08× |
| **A07** | **learner CPU** | **1 331** | **1,16×** |
| A04 | `envs/runner=4` | 1 303 | 1,13× |
| A03 | `workers=8` | 1 203 | 1,05× |
| A05 | `envs/runner=8` | 1 204 | 1,05× |
| A12 | `w8 + envs/runner=8` | 1 195 | 1,04× |
| **A00** | **référence** (batch 1024, mb 128, w6, GPU) | **1 148** | 1,00× |
| A06 | `envs/runner=16` | 1 144 | 1,00× |
| A10 | `batch=4096, mb=128` | 1 176 | 1,02× |
| A02 | `workers=4` | 1 032 | 0,90× |
| A01 | `workers=2` | 754 | 0,66× |

Les deux seuls leviers qui rapportent vraiment du débit (batch, minibatch) sont
exactement ceux qui détruisent la convergence. Les leviers purs plafonnent à
+13 %.

---

## Trois recommandations corrigées par la mesure

### 1. Le GPU est contre-productif sur cette configuration

**Recommandé avant** : `num_gpus_per_learner=1`.
**Mesuré** : learner CPU = 1 331 éch/s contre 1 148 sur GPU, soit **+16 % sans
GPU**.

À batch 1024 et minibatch 128, chaque gradient step est minuscule : le coût de
lancement des kernels et les transferts CPU↔GPU dépassent le calcul. Le CPU
garde ces tenseurs en cache.

Mes deux recommandations étaient incohérentes entre elles : je conseillais un
petit batch **et** le GPU, alors que le petit batch est précisément ce qui rend
le GPU inutile. Le ×1,9 invoqué venait d'une mesure à batch 16384.

**Le GPU redevient indispensable si le réseau grossit** : ×16,6 mesuré sur
[2048,2048] (cf. [RESULTS_GPU_CLUSTER.md](RESULTS_GPU_CLUSTER.md)). Le seuil de
bascule est entre 512 et 2048 de largeur.

### 2. `num_envs_per_env_runner` ne sert à rien ici

**Recommandé avant** : « ton meilleur pari sur 8 cœurs », sur la foi d'un ×1,63
mesuré sur 128 cœurs à batch 16384.
**Mesuré sous contrainte** : ×1,13 au mieux (4 envs), et rien au-delà.

Et surtout, test de pureté sur 3 graines :

| config | pas → seuil | temps → seuil | AUC (±σ) |
|---|---:|---:|---:|
| E1 `envs/runner=1` | 35 157 | **31,3 s** | **-523 ± 10** |
| E2 `envs/runner=4` | 41 301 | 32,1 s | -633 ± 96 |
| E3 `envs/runner=16` | 38 229 | 31,6 s | -633 ± 59 |

**Les trois arrivent au seuil en 31-32 s.** Le +13 % de débit de E2 est
exactement annulé par ses +17 % d'échantillons.

Sur la pureté stricte, le verdict est nuancé : la dégradation d'AUC fait 1 à 2 σ
selon la config — pas concluant à 3 graines. Mais **la variabilité explose**
(σ de 10 pour E1 contre 96 et 59), ce qui est cohérent avec le mécanisme
suspecté : plus d'envs parallèles = plus d'épisodes tronqués aux frontières de
batch. Le levier ne casse pas l'apprentissage, il ne rapporte simplement rien
et coûte en reproductibilité.

### 3. Le batch se comporte comme sur le cluster, pas comme sur Jetson

**Prédit** : +12 % de débit en augmentant le batch (régime Jetson, CPU-pauvre).
**Mesuré** : +108 % de 1024 à 4096, +213 % à 16384.

L'erreur venait d'une comparaison biaisée : le +12 % du Jetson portait sur
4096→16384, alors que je l'appliquais à 1024→4096. Ce n'est pas le nombre de
cœurs qui commande, c'est la plage de batch.

Ça resserre l'arbitrage final sans le renverser — le petit batch reste devant.

---

## Configuration recommandée (révisée, cohérente)

```python
.env_runners(
    num_env_runners=6,            # sature à 6-8 ; 2 ou 4 coûtent cher
    num_envs_per_env_runner=1,    # mesuré sans gain net sur cette config
    num_gpus_per_env_runner=0,    # jamais
)
.learners(
    num_learners=0,
    num_gpus_per_learner=0,       # CPU +16 % ICI ; repasser à 1 si réseau > ~512 de large
)
.training(
    train_batch_size_per_learner=1024,
    minibatch_size=128,
    num_epochs=6,
)
```

Temps estimé pour atteindre le seuil : **~30 s** (39 936 échantillons à
1 331 éch/s), contre ~48 s pour la baseline batch 4096.

Combinaisons dominées, mesurées :

| config | temps estimé |
|---|---:|
| **batch 1024, mb 128, learner CPU** | **~30 s** |
| batch 1024, mb 128, learner GPU | ~35 s |
| batch 4096, mb 512 | ~48 s |
| batch 4096, mb 128 | ~52 s |
| batch 16384 | inutilisable (2 graines sur 3 censurées) |

**L'A40 resterait inutilisé sur ce workload.** C'est le résultat, pas un
renoncement : sur un MLP 256×256 à petit batch, il n'a rien à calculer qui
justifie le coût des transferts. Il redevient décisif dès que le réseau
grossit.

## Limites

- Émulation, pas la vraie VM (cf. plus haut).
- Les temps → seuil combinent l'efficacité-échantillon (mesurée ici) et le
  débit (mesuré ici aussi) : cohérents, mais les runs durent 30 s, où le
  démarrage de Ray et la variance de graine pèsent lourd.
- 3 graines ; le bruit inter-graines n'est pas uniforme entre configs.
- 52 % des itérations n'ont aucun épisode terminé (batch 1024, épisodes de
  200 pas) : `episode_return_mean` y est absent et le notebook l'interpole par
  graine. Un budget de batch plus grand donnerait des courbes moins trouées.
- Une seule tâche (`MultiAgentPendulum`, 2 agents) et un seul réseau.
