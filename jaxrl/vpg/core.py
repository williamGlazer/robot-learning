import functools

import jax
from jax import numpy as jnp
from gym.spaces import Box, Discrete

import scipy

import haiku as hk
from distrax import Categorical, MultivariateNormalDiag


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if jnp.isscalar(shape) else (length, *shape)


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(hk.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def __call__(self, obs):
        return self._distribution(obs)


class MLPCategoricalActor(Actor):
    def __init__(self, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = hk.nets.MLP(
            list(hidden_sizes) + [act_dim], activation=activation
        )
        raise NotImplementedError("Not properly tested since didn't know if required")

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)


class MLPGaussianActor(Actor):
    def __init__(self, act_dim, hidden_sizes, activation):
        super().__init__()
        self.act_dim = act_dim
        self.mu_net = hk.nets.MLP(list(hidden_sizes) + [act_dim], activation=activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        log_std = hk.get_parameter(
            "log_std", [self.act_dim], init=hk.initializers.Constant(-0.5)
        )
        std = jnp.exp(log_std)
        return MultivariateNormalDiag(mu, std)


class MLPCritic(hk.Module):
    def __init__(self, hidden_sizes, activation):
        super().__init__()
        self.v_net = hk.nets.MLP(list(hidden_sizes) + [1], activation=activation)

    def __call__(self, obs):
        return self.v_net(obs).squeeze(axis=-1)


class MLPActorCritic:
    def __init__(
        self,
        action_space,
        rng: jnp.ndarray,
        sample_state: jnp.ndarray,
        hidden_sizes=(64, 64),
        activation=jax.nn.tanh,
    ):
        self.action_space = action_space
        (act_dim,) = action_space.shape  # unpack tuple dimension
        actor_rng, critic_rng = jax.random.split(rng)

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = hk.transform(
                lambda x: MLPGaussianActor(act_dim, hidden_sizes, activation)(x)
            )
        elif isinstance(action_space, Discrete):
            self.pi = hk.transform(
                lambda x: MLPCategoricalActor(act_dim, hidden_sizes, activation)(x)
            )
        self.pi_params = self.pi.init(actor_rng, sample_state)

        # build value function
        self.v = hk.transform(lambda x: MLPCritic(hidden_sizes, activation)(x))
        self.v_params = self.v.init(critic_rng, sample_state)

    @functools.partial(jax.jit, static_argnums=0)
    def forward(self, pi_params, v_params, obs, rng: jnp.ndarray):
        actor_rand, critic_rand, sample_rand = jax.random.split(rng, num=3)

        pi = self.pi.apply(pi_params, x=obs, rng=actor_rand)
        a = pi.sample(seed=sample_rand)
        logp_a = self._log_prob_from_distribution(pi=pi, act=a)
        v = self.v.apply(v_params, x=obs, rng=critic_rand)

        return a, v, logp_a

    @functools.partial(jax.jit, static_argnums=0)
    def _log_prob_from_distribution(self, pi, act):
        if isinstance(self.action_space, Box):
            return pi.log_prob(act)
        elif isinstance(self.action_space, Discrete):
            raise NotImplementedError
