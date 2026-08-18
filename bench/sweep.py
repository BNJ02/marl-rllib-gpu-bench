"""Sweep de convergence : 8 configs autour d'une baseline, un levier à la fois.

Le bench v1 (`bench/configs.py`) ne mesure qu'un axe : le débit. Insuffisant —
augmenter le batch améliore le débit mais réduit le nombre de gradient steps
par échantillon, donc potentiellement la vitesse de convergence. Une config
"2× plus rapide" qui a besoin de 3× plus d'échantillons est une régression.

Ce module sert à classer chaque levier RLlib en trois catégories :
  - **pur** : accélère sans toucher aux maths de l'optimisation (la courbe
    retour-vs-échantillons se superpose à la baseline) ;
  - **compromis gagnant** : dégrade l'efficacité-échantillon, mais le gain en
    débit fait que le seuil est atteint plus vite *en temps mural* ;
  - **compromis perdant** : la dégradation n'est pas compensée.

Env : `MultiAgentPendulum` (2 agents). CartPole (utilisé par le v1) sature en
~20 itérations ; à batch 32768 ça ferait 2 itérations, aucune résolution pour
mesurer une dégradation de convergence. Pendulum a une courbe longue et lisse.

Piège Ray déjà rencontré sur ce projet : les builders doivent être des
fonctions top-level d'un module importable. Un `sys.path.insert()` + import
dynamique fait planter les `EnvRunner` distants (`ModuleNotFoundError` côté
acteur, le sys.path du driver n'est pas hérité).
"""

from __future__ import annotations

from dataclasses import dataclass

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.connectors.env_to_module import MeanStdFilter
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.examples.envs.classes.multi_agent import MultiAgentPendulum

NUM_AGENTS = 2
SHARED_POLICY = "shared"

# Baseline : le point autour duquel on fait varier un seul facteur à la fois.
BASE_BATCH = 4096
BASE_MINIBATCH = 512
BASE_WORKERS = 4


@dataclass
class SweepSpec:
    name: str
    lever: str          # le levier isolé par rapport à la baseline
    hypothesis: str     # pur / non-pur, prédit AVANT de mesurer
    batch: int = BASE_BATCH
    minibatch: int = BASE_MINIBATCH
    workers: int = BASE_WORKERS
    num_learners: int = 0
    use_gpu: bool = True


# S1-S4 : ratio batch/minibatch CONSTANT (=8), donc nombre de gradient steps
# par échantillon fixe -> isole l'effet de la TAILLE de batch (bruit du
# gradient) de l'effet du NOMBRE d'updates.
# S5-S6 : l'inverse, batch fixe et ratio variable -> isole le nombre d'updates.
# Sans ces deux familles séparées, les deux effets restent confondus.
SWEEP: dict[str, SweepSpec] = {
    "S1": SweepSpec("S1", "batch=1024", "non pur : meilleure eff.-éch., débit moindre",
                    batch=1024, minibatch=128),
    "S2": SweepSpec("S2", "baseline", "référence"),
    "S3": SweepSpec("S3", "batch=16384", "non pur : dégradation attendue",
                    batch=16384, minibatch=2048),
    "S4": SweepSpec("S4", "batch=65536", "non pur : dégradation forte",
                    batch=65536, minibatch=8192),
    "S5": SweepSpec("S5", "minibatch=128", "non pur : 8x plus de gradient steps",
                    minibatch=128),
    "S6": SweepSpec("S6", "minibatch=4096", "non pur : 1 seul step par epoch",
                    minibatch=4096),
    "S7": SweepSpec("S7", "num_env_runners=1", "PUR attendu : mêmes maths",
                    workers=1),
    # num_learners=1 (pas 2) : `train_batch_size_per_learner` est PAR learner,
    # donc num_learners=2 doublerait silencieusement le batch effectif et le
    # test ne serait plus pur par construction (cf. SOTA_MARL_RLlib
    # docs/07-scaling-et-perf.md §7.1). Ici : un seul learner, mais dans un
    # acteur Ray séparé du driver. Test de PLACEMENT, pas de parallélisme.
    "S8": SweepSpec("S8", "num_learners=1", "PUR attendu : batch effectif inchangé",
                    num_learners=1, workers=BASE_WORKERS - 1),
    # Rattrapage de S4, qui fait OOM à 4 workers sur les 8 Go de mémoire unifiée
    # du Jetson (EnvRunners tués, 0 échantillon collecté). Moins de workers = moins
    # de process torch simultanés.
    #
    # Pourquoi c'est valide malgré le changement de deux paramètres à la fois :
    # S7 a mesuré que `num_env_runners` est PUR (+3.8 % d'échantillons vs une
    # variance inter-graines de 4 % sur la baseline). Sa courbe
    # retour-vs-échantillons est donc comparable aux autres configs. Son TEMPS
    # mural, lui, ne l'est pas — à ne pas mettre dans le classement chrono.
    "S4b": SweepSpec("S4b", "batch=65536 (2 workers)",
                     "comme S4, mais mesurable : convergence comparable, chrono non",
                     batch=65536, minibatch=8192, workers=2),
}


def build_sweep_config(spec: SweepSpec, seed: int, force_cpu: bool = False) -> PPOConfig:
    """`PPOConfig` pour une spec + une graine.

    Hyperparamètres Pendulum, à ne pas confondre avec ceux de CartPole (v1) :
    - `MeanStdFilter` : sans normalisation d'observation, Pendulum apprend mal.
    - `vf_clip_param=200` : les retours vont à ~-1600 par agent ; le défaut
      (10) écrêterait 100 % de la value loss — piège déjà rencontré et
      documenté dans simple_rl/train.py (critique incapable de fitter, donc
      avantages faux et policy qui s'effondre).
    - `gamma=0.95`, `lambda_=0.5` : valeurs usuelles PPO sur Pendulum.
    """
    return (
        PPOConfig()
        .environment(MultiAgentPendulum, env_config={"num_agents": NUM_AGENTS})
        .debugging(seed=seed)
        .multi_agent(
            policies={SHARED_POLICY},
            policy_mapping_fn=lambda agent_id, episode, **kw: SHARED_POLICY,
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={SHARED_POLICY: RLModuleSpec()},
            ),
        )
        .env_runners(
            num_env_runners=spec.workers,
            # multi_agent=True obligatoire : sans ça le filtre reçoit l'espace
            # d'observation multi-agent (dict AgentID -> Box) et essaie d'en
            # faire un seul np.zeros(shape) -> TypeError au démarrage.
            env_to_module_connector=(
                lambda env, spaces=None, device=None: MeanStdFilter(multi_agent=True)
            ),
        )
        .training(
            train_batch_size_per_learner=spec.batch,
            minibatch_size=spec.minibatch,
            num_epochs=6,
            lr=3e-4,
            gamma=0.95,
            lambda_=0.5,
            vf_clip_param=200.0,
        )
        .learners(
            num_learners=spec.num_learners,
            num_gpus_per_learner=1 if (spec.use_gpu and not force_cpu) else 0,
        )
    )
