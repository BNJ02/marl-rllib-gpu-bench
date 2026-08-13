"""Les 5 configs PPO à comparer, CPU vs GPU, sur `MultiAgentCartPole`.

Même env et mêmes hyperparams `training()` que le bench Jetson
(SOTA_MARL_RLlib/docs/jetson_gpu_optim.md) pour que les débits restent
comparables d'une machine à l'autre — seul ce qui varie explicitement dans le
tableau ci-dessous change.

Piège évité (rencontré sur Jetson) : ces builders sont des fonctions
top-level d'un module réellement importable. Un `sys.path.insert()` +
`importlib.import_module()` fait planter les `EnvRunner`s distants
(`ModuleNotFoundError` côté acteur Ray, le sys.path modifié côté driver n'est
pas hérité par les process qu'il spawn).

| # | batch | minibatch | workers | GPU | learners | réseau     |
|---|------:|----------:|--------:|-----|----------|------------|
| A |  2048 |       256 |       6 | non | 0        | défaut     |
| B |  2048 |       256 |       6 | oui | 0        | défaut     |
| C | 32768 |      4096 |       6 | oui | 0        | défaut     |
| D | 32768 |      4096 |       6 | oui | 0        | [512, 512] |
| E | 32768 |      4096 |       5 | oui | 1        | défaut     |

`workers` = cœurs−2 par défaut (6 sur une VM 8 cœurs) ; ajustable en CLI.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.examples.envs.classes.multi_agent import MultiAgentCartPole

NUM_AGENTS = 2
SHARED_POLICY = "shared"


@dataclass
class ConfigSpec:
    """Description d'une config, indépendante du nombre de cœurs de la
    machine qui l'exécute (`workers` est résolu à l'appel, cf. `resolve()`)."""

    name: str
    label: str
    batch: int
    minibatch: int
    use_gpu: bool
    num_learners: int = 0
    hiddens: list[int] | None = None
    # workers=None -> résolu par l'appelant (cœurs-2, ou cœurs-3 pour E qui
    # réserve un cœur de plus au process driver quand le learner est un
    # acteur Ray séparé).
    workers_offset: int = 2

    def resolve_workers(self, cpu_count: int) -> int:
        return max(1, cpu_count - self.workers_offset)


CONFIGS: dict[str, ConfigSpec] = {
    "A": ConfigSpec("A", "référence CPU", batch=2048, minibatch=256, use_gpu=False),
    "B": ConfigSpec("B", "GPU petit batch", batch=2048, minibatch=256, use_gpu=True),
    "C": ConfigSpec("C", "GPU gros batch", batch=32768, minibatch=4096, use_gpu=True),
    "D": ConfigSpec(
        "D", "GPU gros réseau", batch=32768, minibatch=4096, use_gpu=True,
        hiddens=[512, 512],
    ),
    "E": ConfigSpec(
        "E", "learner distant", batch=32768, minibatch=4096, use_gpu=True,
        num_learners=1, workers_offset=3,
    ),
}


def build_config(spec: ConfigSpec, cpu_count: int, force_cpu: bool = False) -> PPOConfig:
    """Construit la `PPOConfig` pour `spec`, résolue pour une machine à
    `cpu_count` cœurs. `force_cpu=True` ignore `spec.use_gpu` (smoke-test sans
    GPU) sans changer le reste de la config."""
    workers = spec.resolve_workers(cpu_count)
    use_gpu = spec.use_gpu and not force_cpu

    rl_module_kwargs = {}
    if spec.hiddens:
        rl_module_kwargs["model_config"] = {"fcnet_hiddens": spec.hiddens}

    return (
        PPOConfig()
        .environment(MultiAgentCartPole, env_config={"num_agents": NUM_AGENTS})
        .multi_agent(
            policies={SHARED_POLICY},
            policy_mapping_fn=lambda agent_id, episode, **kw: SHARED_POLICY,
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={SHARED_POLICY: RLModuleSpec()},
            ),
            **rl_module_kwargs,
        )
        .training(
            train_batch_size_per_learner=spec.batch,
            minibatch_size=spec.minibatch,
            num_epochs=6,
            lr=3e-4,
        )
        .env_runners(num_env_runners=workers)
        .learners(
            num_learners=spec.num_learners,
            num_gpus_per_learner=1 if use_gpu else 0,
        )
    )
