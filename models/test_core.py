import unittest

import jax
import haiku as hk
import numpy as np
from gym.spaces import Box, Discrete
from jax import numpy as jnp

from core import MLPGaussianActor, MLPCategoricalActor, MLPCritic, MLPActorCritic
from distrax import Categorical, MultivariateNormalDiag


class TestActor(unittest.TestCase):
    obs_dim = 4
    act_dim = 2
    hidden_sizes = [50, 25]
    seed = 42

    def test_gaussian(self):
        args = dict(act_dim=self.act_dim, hidden_sizes=self.hidden_sizes, activation=jax.nn.tanh)
        obs = jnp.zeros(self.obs_dim, dtype=jnp.float32)
        rng = jax.random.PRNGKey(self.seed)

        actor = hk.transform(lambda *x: MLPGaussianActor(**args)(*x))
        actor_params = actor.init(rng, obs)
        pi = actor.apply(actor_params, rng, obs)

        self.assertIsInstance(pi, MultivariateNormalDiag)

    def test_logp_gaussian(self):
        args = dict(act_dim=self.act_dim, hidden_sizes=self.hidden_sizes, activation=jax.nn.tanh)
        obs = jnp.zeros(self.obs_dim, dtype=jnp.float32)
        act = jnp.zeros(self.act_dim, dtype=jnp.float32)
        rng = jax.random.PRNGKey(self.seed)

        actor = hk.transform(lambda x: MLPGaussianActor(**args)(x))
        actor_params = actor.init(rng, obs)
        pi = actor.apply(actor_params, rng, obs)

        self.assertIsInstance(pi, MultivariateNormalDiag)

    def test_categorical(self):
        args = dict(act_dim=self.act_dim, hidden_sizes=self.hidden_sizes, activation=jax.nn.tanh)
        obs = jnp.zeros(self.obs_dim, dtype=jnp.float32)
        rng = jax.random.PRNGKey(self.seed)

        actor = hk.transform(lambda x: MLPCategoricalActor(**args)(x))
        actor_params = actor.init(rng, obs)
        pi = actor.apply(actor_params, rng, obs)

        self.assertIsInstance(pi, Categorical)


class TestCritic(unittest.TestCase):
    obs_dim = 4
    hidden_sizes = [50, 25]
    seed = 42

    def test_critic(self):
        args = dict(hidden_sizes=self.hidden_sizes, activation=jax.nn.tanh)
        obs = jnp.zeros(self.obs_dim, dtype=jnp.float32)
        rng = jax.random.PRNGKey(self.seed)

        critic = hk.transform(lambda x: MLPCritic(**args)(x))
        actor_params = critic.init(rng, obs)
        pi = critic.apply(actor_params, rng, obs)

        self.assertIsInstance(pi, jnp.ndarray)


class TestActorCritic(unittest.TestCase):
    obs_dim = 4
    hidden_sizes = [50, 25]
    seed = 42

    def test_continuous_forward(self):
        action_space = Box(low=0, high=1, shape=[10])
        actor_rng = jax.random.PRNGKey(self.seed)
        critic_rng = jax.random.PRNGKey(self.seed)
        forward_rng = jax.random.PRNGKey(self.seed)
        obs = jnp.zeros(self.obs_dim, dtype=jnp.float32)
        model = MLPActorCritic(
            action_space=action_space,
            actor_rng=actor_rng,
            critic_rng=critic_rng,
            sample_state=obs
        )

        action, value, log_p_action = model.forward(obs, forward_rng)

        self.assertEquals(action.shape, action_space.shape)
        self.assert_((action >= action_space.low).all())
        self.assert_((action >= action_space.high).all())

        self.assertEquals(value.size, 1)

        self.assertEquals(log_p_action.size, 1)


    def test_batched_forward(self):
        action_space = Box(low=0, high=1, shape=[10])
        actor_rng = jax.random.PRNGKey(self.seed)
        critic_rng = jax.random.PRNGKey(self.seed)
        forward_rng = jax.random.PRNGKey(self.seed)
        obs = jnp.zeros([32, self.obs_dim], dtype=jnp.float32)
        model = MLPActorCritic(
            action_space=action_space,
            actor_rng=actor_rng,
            critic_rng=critic_rng,
            sample_state=obs
        )

        action, value, log_p_action = model.forward(obs, forward_rng)

        self.assertEquals(action.shape, (32, action_space.shape[0]))
        self.assertEquals(value.size, 32)
        self.assertEquals(log_p_action.size, 32)


if __name__ == '__main__':
    unittest.main()
