import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp

import haiku as hk
from gym.spaces import Box


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if jnp.isscalar(shape) else (length, *shape)


def count_vars(params):
    return sum(jax.tree_leaves(jax.tree_map(lambda x: x.size, params)))


class MLPActor(hk.Module):

    def __init__(self, act_dim: int, hidden_sizes: list[int], activation: callable, act_limit: float):
        super().__init__()
        self.pi = hk.nets.MLP(output_sizes=list(hidden_sizes) + [act_dim], activation=activation)
        self.act_limit = act_limit

    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        # Return output from network scaled to action space limits.
        return self.act_limit * jax.nn.tanh(self.pi(obs))


class MLPQFunction(hk.Module):

    def __init__(self, hidden_sizes: list[int], activation: callable):
        super().__init__()
        self.q = hk.nets.MLP(output_sizes=list(hidden_sizes) + [1], activation=activation)

    def __call__(self, obs: jnp.array, act: jnp.array) -> jnp.array:
        q = self.q(jnp.concatenate([obs, act], axis=-1))
        return jnp.squeeze(q, -1)  # Critical to ensure q has right shape.


class ACParams(NamedTuple):
    pi: hk.Params
    q1: hk.Params
    q2: hk.Params


class MLPActorCritic:

    def __init__(
            self,
            sample_state: jnp.ndarray,
            sample_action: jnp.ndarray,
            rng: jnp.ndarray,
            action_space: Box,
            hidden_sizes=(256, 256),
            activation=jax.nn.tanh
    ):
        super().__init__()

        rng, q1_rng, q2_rng = jax.random.split(rng, num=3)

        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi, pi_params = self._init_hk_transform(
            module=MLPActor, args=(act_dim, hidden_sizes, activation, act_limit),
            sample_input=sample_state, rng=rng
        )
        self.q1, q1_params = self._init_hk_transform(
            module=MLPQFunction, args=(hidden_sizes, activation),
            sample_input=(sample_state, sample_action), rng=q1_rng
        )
        self.q2, q2_params = self._init_hk_transform(
            module=MLPQFunction, args=(hidden_sizes, activation),
            sample_input=(sample_state, sample_action), rng=q2_rng
        )

        self.params = ACParams(pi=pi_params, q1=q1_params, q2=q2_params)

    def _init_hk_transform(self, module: type, args: tuple, sample_input: tuple[jnp.ndarray, jnp.ndarray] | jnp.ndarray, rng: jnp.ndarray) -> tuple[hk.Transformed, hk.Params]:
        if type(sample_input) == tuple:
            transform = hk.transform(lambda obs, act: module(*args)(obs, act))
            params = transform.init(rng, *sample_input)
        else:
            transform = hk.transform(lambda obs: module(*args)(obs))
            params = transform.init(rng, sample_input)
        return transform, params

    @functools.partial(jax.jit, static_argnums=0)
    def __call__(self, pi_params: hk.Params, rng: jnp.ndarray, obs: jnp.ndarray):
        return self.pi.apply(params=pi_params, rng=rng, obs=obs)

    def act(self, obs: jnp.ndarray, rng: jnp.ndarray):
        return self(pi_params=self.params.pi, rng=rng, obs=obs)

    def set_params(self, pi: hk.Params, q1: hk.Params, q2: hk.Params):
        self.params = ACParams(pi=pi, q1=q1, q2=q2)
