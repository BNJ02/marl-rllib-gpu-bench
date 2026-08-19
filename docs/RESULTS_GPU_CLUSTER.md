# Résultats sur cluster GPU (dataia25)

Machine : 128 cœurs, 119 Go RAM, **RTX 4090 + RTX 3090** (hétérogènes),
Ray 2.57, torch 2.13+cu130, `OMP_NUM_THREADS=1`.
Même env que le sweep Jetson : `MultiAgentPendulum` (2 agents), PPO, seuil -800.

Deux campagnes :
- **matrice de débit** — 25 configs, 5 itérations, un levier à la fois ;
- **sweep de convergence** — 8 configs × 3 graines × 300k pas.

49 runs, 0 échec. Les deux configs que le Jetson ne pouvait pas exécuter
(S4 : OOM ; S8 : `torch.distributed` absent du wheel JetPack) tournent ici.

---

## 1. Matrice de débit — les 10 leviers

| cfg | levier | éch/s | vs réf |
|---|---|---:|---:|
| **M24** | **w64 + batch 65536 + GPU** | **16 862** | **2,78×** |
| M06 | `num_envs_per_env_runner=16` | 9 879 | 1,63× |
| M05 | `num_envs_per_env_runner=4` | 9 048 | 1,49× |
| M04 | `num_env_runners=64` | 9 001 | 1,48× |
| M20 | `minibatch=16384` (1 step/epoch) | 8 485 | 1,40× |
| M03 | `num_env_runners=32` | 8 349 | 1,38× |
| M17 | `batch=65536` | 8 068 | 1,33× |
| M18 | `batch=262144` | 7 451 | 1,23× |
| M11 | `num_learners=2` (batch eff. ×2) | 7 115 | 1,17× ⚠ |
| M13 | `num_cpus_per_learner=8` | 6 159 | 1,02× |
| M21 | réseau [512,512] | 6 145 | 1,01× |
| **M00** | **référence** (batch 16384, mb 2048, w16, GPU) | **6 068** | **1,00×** |
| M10 | `num_learners=1` | 5 912 | 0,97× |
| M12 | `num_learners=2`, batch compensé | 5 847 | 0,96× |
| M15 | `evaluation_num_env_runners=16` | 5 663 | 0,93× |
| M14 | `evaluation_num_env_runners=4` | 4 709 | 0,78× |
| M02 | `num_env_runners=8` | 3 887 | 0,64× |
| M09 | learner sur CPU | 3 170 | 0,52× |
| M16 | `batch=4096` | 2 975 | 0,49× |
| M08 | `num_gpus_per_env_runner=0.05` | 2 745 | 0,45× |
| M01 | `num_env_runners=4` | 2 325 | 0,38× |
| M22 | réseau [2048,2048] | 2 058 | 0,34× |
| M19 | `minibatch=256` | 1 864 | 0,31× |
| M23 | réseau [2048,2048], learner CPU | 124 | 0,02× |

### L'accélération GPU dépend de la taille du modèle, pas de la machine

| réseau | learner GPU | learner CPU | gain GPU |
|---|---:|---:|---:|
| [256,256] | 6 068 | 3 170 | **×1,9** |
| [2048,2048] | 2 058 | 124 | **×16,6** |

C'est ce qui réconcilie deux mesures qui semblaient se contredire : sur Jetson
le GPU n'apportait que 0-7 % ([v1](jetson_gpu_optim.md)), ici ×1,9, et ×16,6
sur un gros réseau. À petit modèle on ne paie que des frais fixes (lancement de
kernels, transferts) ; à gros modèle le calcul domine et le GPU écrase le CPU.

### Trois pièges que le débit seul aurait fait tomber

**1. Le bi-GPU ne sert à rien ici.** `num_learners=2` affiche +17 % — mais
`train_batch_size_per_learner` est **par learner**, donc le batch effectif a
doublé. À batch effectif égal (M12) : **0,96×**, une perte. Le second GPU
n'apporte rien tant que chaque learner n'a pas de quoi le saturer, et
l'hétérogénéité 3090/4090 fait attendre le plus rapide.

**2. GPU sur les env_runners : 0,45×.** Le levier le plus nuisible du lot. Le
réseau d'inférence est minuscule : on paie un contexte CUDA par runner et des
transferts pour un forward négligeable. Confirme §7.5 de
[SOTA_MARL_RLlib](https://github.com/BNJ02/marl-rllib-sota) : **GPU sur le
learner, jamais sur les runners**.

**3. Sous-dimensionner l'évaluation coûte plus que la sur-dimensionner** :
0,78× à 4 workers contre 0,93× à 16. L'éval est un travail fixe qui bloque
l'itération ; mal parallélisée, elle reste sur le chemin critique.

### Un résultat contraire à la documentation

**`num_envs_per_env_runner` fonctionne en multi-agent** : ×1,63 à 16 envs par
runner, soit mieux que 64 workers (×1,48) avec **4× moins de process**. Or la
doc RLlib citée en §7.2 de SOTA_MARL_RLlib affirme les envs multi-agents non
vectorisables et ce levier sans effet en MARL.

Vérifié non-artefact : les configs collectent toutes exactement 16 384 pas par
itération, même `num_agent_steps_sampled_lifetime`.

⚠ **Sa pureté n'est pas établie** : à 256 envs parallèles (16×16), M06 ne
termine que 256 épisodes contre 400 pour la référence — beaucoup plus
d'épisodes à cheval sur les frontières de batch, donc un signal d'apprentissage
potentiellement différent. Gain de débit certain, effet sur la convergence non
mesuré.

### Saturation de l'échantillonnage

4 → 8 : ×1,67 · 8 → 16 : ×1,56 · 16 → 32 : ×1,38 · **32 → 64 : ×1,08**

L'échantillonnage sature vers 32 workers : à batch 16384 et 64 workers, chaque
worker ne collecte que 256 pas par itération et la synchronisation domine.

---

## 2. Sweep de convergence — 8 configs × 3 graines

Médianes sur 3 graines :

| # | Levier | Éch. → seuil | Temps → seuil | Verdict |
|---|---|---:|---:|---|
| **S1** | `batch=1024` | **39 936** (0,35×) | **45 s** (0,71×) | **gain net** |
| S2 | baseline | 114 688 | 63 s | référence |
| S3 | `batch=16384` | 2/3 censurées | 123 s | compromis perdant |
| S4 | `batch=65536` | jamais (3/3) | — | ne converge pas |
| S5 | `minibatch=128` | 61 440 (0,54×) | 72 s (**1,14×**) | **perd en temps** |
| S6 | `minibatch=4096` | jamais (3/3) | — | ne converge pas |
| S7 | `num_env_runners=1` | 110 592 (0,96×) | 176 s (2,79×) | **pur** |
| S8 | `num_learners=1` | 114 688 (1,00×) | 78 s (1,24×) | pur, mais coûteux |

### `num_env_runners` est pur — reproduit sur deux architectures

| machine | écart d'échantillons vs baseline | gain en temps |
|---|---:|---:|
| Jetson (ARM, 6 cœurs) | +1,3 % | 2,36× |
| dataia25 (x86, 128 cœurs) | **-3,1 %** | 2,75× |

Le signe de l'écart **s'inverse** entre les deux plateformes : signature d'un
effet nul noyé dans le bruit, pas d'un biais systématique. C'est une
confirmation plus forte que si le même +1,3 % s'était répété. **Seul levier de
toute l'étude qu'on peut pousser sans arbitrage.**

### La recommandation s'inverse selon le matériel

C'est le résultat le plus actionnable de la campagne :

| machine | meilleur levier | pourquoi |
|---|---|---|
| CPU (Jetson) | baisser `minibatch_size` | pas de coût de lancement de kernels |
| GPU + gros CPU | baisser `train_batch_size` | chaque gradient step porte un coût GPU fixe |

Sur Jetson, S5 (`minibatch=128`) gagnait en temps (0,82×). Ici il **perd**
(1,14×) : multiplier les petits updates multiplie les lancements de kernels et
les synchronisations CPU↔GPU. La matrice le mesure isolément — `minibatch=256`
tombe à 0,31× de débit.

S1 (`batch=1024`) gagne en revanche sur les deux axes (0,35× d'échantillons,
0,71× de temps) : il réduit l'inefficacité-échantillon sans multiplier le
nombre d'updates.

### Le gros batch : aucun régime gagnant

| batch | débit | convergence |
|---|---:|---|
| 1 024 | 0,49× | 39 936 pas |
| 4 096 | 1,00× | 114 688 pas |
| 16 384 | 1,33× | 2 graines sur 3 censurées |
| 65 536 | 1,33× | **jamais** (3/3, niveau aléatoire) |
| 262 144 | 1,23× | non testé (inutile) |

S4 était la case manquante du rapport Jetson (« non mesurable, OOM »). Mesurée
ici : **-2 993 de retour final, niveau aléatoire**. Et au-delà de 65 536, même
l'argument du débit s'effondre (262 144 est plus lent que 65 536).

C'est la démonstration la plus nette que le débit seul induit en erreur : M17
(batch 65536) est dans le tiers supérieur de la matrice de débit, et
n'apprend rien.

---

## 3. Configuration recommandée sur cette machine

```python
.env_runners(
    num_env_runners=32,          # sature vers 32 ; 64 n'apporte que +8%
    num_envs_per_env_runner=16,  # x1,63 — mais vérifier la convergence (voir ci-dessus)
    num_gpus_per_env_runner=0,   # JAMAIS de GPU sur les runners (0,45x)
)
.learners(
    num_learners=0,              # 1 coûte 15%, 2 ne rapporte rien à batch égal
    num_gpus_per_learner=1,      # x1,9 ici, x16,6 sur gros réseau
)
.training(
    train_batch_size_per_learner=1024,   # petit batch : gagne sur les deux axes
    minibatch_size=128,                  # ratio 8 ; NE PAS descendre plus bas sur GPU
)
```

M24 (`w64 + batch 65536 + GPU`) atteint 16 862 éch/s, 2,78× la référence — et
reste **le pire choix pratique**, puisque batch 65536 ne converge jamais. À
n'utiliser que comme borne supérieure de débit, pas comme configuration.

## Limites

- Une tâche, un réseau (MLP par défaut sauf M21-M23). Le point d'équilibre
  débit / efficacité-échantillon dépend du coût de simulation de l'env.
- Les runs de convergence durent 45-80 s sur cette machine : le démarrage de
  Ray (~15 s) et la variance de graine pèsent lourd dans la comparaison. Pour
  départager des configs à moins de 20 % d'écart, il faudrait un budget de pas
  plus grand ou davantage de graines.
- 3 graines, et le bruit inter-graines n'est pas uniforme entre configs.
- Machine partagée : 4 utilisateurs connectés pendant une partie de la
  campagne (charge observée 12/128, donc impact négligeable, mais non nul).
- La pureté de `num_envs_per_env_runner` reste à mesurer.
