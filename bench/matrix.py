"""Matrice de débit : les 10 leviers RLlib, un facteur à la fois.

Complément du sweep de convergence (`bench/sweep.py`). La séparation vient du
résultat central établi sur Jetson :

  - la courbe retour-vs-échantillons ne dépend PAS du matériel (arithmétique de
    gradient) -> mesurée une fois, sur n'importe quelle machine ;
  - le débit et le temps mural en dépendent entièrement -> à mesurer sur CHAQUE
    machine cible.

Ce module ne mesure donc que la moitié dépendante du matériel, ce qui permet
de balayer beaucoup plus de leviers pour pas cher : 5 itérations par config
suffisent à un débit stable, là où une courbe de convergence demande 300k pas
et 3 graines.

Les leviers qui ne changent PAS les maths (allocation de ressources) n'ont
besoin que de cette moitié-là. Ceux qui les changent (batch, minibatch,
num_learners>=2) sont en plus repris dans le sweep de convergence.

ATTENTION `num_learners>=2` : `train_batch_size_per_learner` est PAR learner,
donc num_learners=2 DOUBLE le batch effectif. Les specs concernées le signalent
et une variante à batch compensé est fournie pour comparer à batch effectif
égal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.connectors.env_to_module import MeanStdFilter
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.examples.envs.classes.multi_agent import MultiAgentPendulum

NUM_AGENTS = 2
SHARED = "shared"


@dataclass
class MatrixSpec:
    """Une config de la matrice. Les valeurs None = défaut RLlib (non forcé)."""

    name: str
    lever: str
    # --- algorithme
    batch: int = 16_384
    minibatch: int = 2_048
    num_epochs: int = 6
    hiddens: list[int] | None = None
    # --- env_runners
    workers: int = 16
    envs_per_runner: int = 1
    cpus_per_runner: int | None = None
    gpus_per_runner: float = 0.0
    # --- learners
    num_learners: int = 0
    cpus_per_learner: int | None = None
    gpus_per_learner: float = 1.0
    # --- evaluation
    eval_workers: int | None = None
    # --- méta
    note: str = ""

    @property
    def effective_batch(self) -> int:
        """Batch réellement vu par update : `_per_learner` x nb de learners."""
        return self.batch * max(1, self.num_learners)


def build_matrix_config(spec: MatrixSpec, seed: int = 0) -> PPOConfig:
    cfg = (
        PPOConfig()
        .environment(MultiAgentPendulum, env_config={"num_agents": NUM_AGENTS})
        .debugging(seed=seed)
        .multi_agent(
            policies={SHARED},
            policy_mapping_fn=lambda agent_id, episode, **kw: SHARED,
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(rl_module_specs={SHARED: RLModuleSpec()}),
            **({"model_config": {"fcnet_hiddens": spec.hiddens}} if spec.hiddens else {}),
        )
        .training(
            train_batch_size_per_learner=spec.batch,
            minibatch_size=spec.minibatch,
            num_epochs=spec.num_epochs,
            lr=3e-4,
            gamma=0.95,
            lambda_=0.5,
            vf_clip_param=200.0,
        )
    )

    runner_kwargs: dict = {
        "num_env_runners": spec.workers,
        "num_envs_per_env_runner": spec.envs_per_runner,
        "num_gpus_per_env_runner": spec.gpus_per_runner,
        # multi_agent=True obligatoire : sinon le filtre reçoit l'espace
        # d'observation multi-agent (dict AgentID -> Box) et plante au démarrage.
        "env_to_module_connector": (
            lambda env, spaces=None, device=None: MeanStdFilter(multi_agent=True)
        ),
    }
    if spec.cpus_per_runner is not None:
        runner_kwargs["num_cpus_per_env_runner"] = spec.cpus_per_runner
    cfg = cfg.env_runners(**runner_kwargs)

    learner_kwargs: dict = {
        "num_learners": spec.num_learners,
        "num_gpus_per_learner": spec.gpus_per_learner,
    }
    if spec.cpus_per_learner is not None:
        learner_kwargs["num_cpus_per_learner"] = spec.cpus_per_learner
    cfg = cfg.learners(**learner_kwargs)

    if spec.eval_workers is not None:
        cfg = cfg.evaluation(
            evaluation_interval=1,
            evaluation_duration=10,
            evaluation_duration_unit="episodes",
            evaluation_num_env_runners=spec.eval_workers,
        )
    return cfg


def _s(name, lever, **kw) -> MatrixSpec:
    return MatrixSpec(name=name, lever=lever, **kw)


# Référence de la matrice : assez gros pour alimenter beaucoup de workers, et
# learner sur GPU (le cas qui nous intéresse sur cette machine).
BASELINE = _s("M00", "référence (batch16384 mb2048 w16 GPU)")

MATRIX: dict[str, MatrixSpec] = {
    "M00": BASELINE,

    # --- num_env_runners : où sature l'échantillonnage sur 128 cœurs ?
    "M01": _s("M01", "workers=4", workers=4),
    "M02": _s("M02", "workers=8", workers=8),
    "M03": _s("M03", "workers=32", workers=32),
    "M04": _s("M04", "workers=64", workers=64),

    # --- num_envs_per_env_runner : la doc RLlib dit les envs multi-agents NON
    # vectorisables ; gain attendu nul, à vérifier plutôt qu'à croire.
    "M05": _s("M05", "envs/runner=4", envs_per_runner=4),
    "M06": _s("M06", "envs/runner=16", envs_per_runner=16),

    # --- num_cpus_per_env_runner : pur (allocation), n'affecte que l'ordonnancement
    "M07": _s("M07", "cpus/runner=2", cpus_per_runner=2),

    # --- num_gpus_per_env_runner : inférence des runners sur GPU (fractionnaire)
    "M08": _s("M08", "gpus/runner=0.05", gpus_per_runner=0.05),

    # --- num_gpus_per_learner : LE test GPU vs CPU pour le learner
    "M09": _s("M09", "learner CPU (gpus/learner=0)", gpus_per_learner=0.0),

    # --- num_learners : 0 = dans le driver, 1 = acteur dédié, 2 = multi-GPU.
    # ATTENTION : à num_learners=2 le batch effectif DOUBLE (32768).
    "M10": _s("M10", "num_learners=1 (acteur dédié)", num_learners=1),
    "M11": _s("M11", "num_learners=2 (2 GPU, batch eff. x2)", num_learners=2,
              note="batch effectif 32768, PAS comparable à débit égal"),
    "M12": _s("M12", "num_learners=2, batch compensé", num_learners=2, batch=8_192,
              minibatch=1_024, note="batch effectif 16384 = référence"),

    # --- num_cpus_per_learner
    "M13": _s("M13", "cpus/learner=8", cpus_per_learner=8),

    # --- evaluation_num_env_runners : coût de l'éval périodique
    "M14": _s("M14", "eval_workers=4", eval_workers=4),
    "M15": _s("M15", "eval_workers=16", eval_workers=16),

    # --- batch : jusqu'où on peut monter avec 119 Go (impossible sur Jetson)
    "M16": _s("M16", "batch=4096", batch=4_096, minibatch=512),
    "M17": _s("M17", "batch=65536", batch=65_536, minibatch=8_192),
    "M18": _s("M18", "batch=262144", batch=262_144, minibatch=32_768),

    # --- minibatch à batch fixe : coût des updates supplémentaires
    "M19": _s("M19", "minibatch=256", minibatch=256),
    "M20": _s("M20", "minibatch=16384 (1 seul step)", minibatch=16_384),

    # --- réseau plus lourd : le GPU devient-il rentable ?
    "M21": _s("M21", "réseau [512,512]", hiddens=[512, 512]),
    "M22": _s("M22", "réseau [2048,2048]", hiddens=[2048, 2048]),
    "M23": _s("M23", "réseau [2048,2048] learner CPU", hiddens=[2048, 2048],
              gpus_per_learner=0.0),

    # --- combinaison la plus prometteuse : beaucoup de workers + gros batch + GPU
    "M24": _s("M24", "w64 + batch65536 + GPU", workers=64, batch=65_536,
              minibatch=8_192),
}
