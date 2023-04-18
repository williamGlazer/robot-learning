import jax
import jax.numpy as jnp

import haiku as hk


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if jnp.isscalar(shape) else (length, *shape)


def count_vars(params):
    return sum(jax.tree_leaves(jax.tree_map(lambda x: x.size, params)))


class MLPActor(hk.Module):

    def __init__(self, act_dim, hidden_sizes, act_limit):
        super().__init__()
        pi_sizes = list(hidden_sizes) + [act_dim]
        self.pi = hk.nets.MLP(pi_sizes, activation=jax.nn.tanh)
        self.act_limit = act_limit

    def __call__(self, obs):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(obs)

class MLPQFunction(hk.Module):

    def __init__(self, hidden_sizes, activation):
        super().__init__()
        self.q = hk.nets.MLP(list(hidden_sizes) + [1], activation=activation)

    def __call__(self, obs, act):
        q = self.q(jnp.concatenate([obs, act], dim=-1))
        return jnp.squeeze(q, -1)  # Critical to ensure q has right shape.

class MLPActorCritic:

    def __init__(
            self,
            sample_state,
            sample_action,
            rng,
            action_space,
            hidden_sizes=(256, 256),
            activation=jax.nn.relu
    ):
        super().__init__()

        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi, self.pi_params = self._init_hk_transform(
            MLPActor,
            (act_dim, hidden_sizes, act_limit),
            (sample_state,),
            rng
        )
        self.q1, self.q1_params = self._init_hk_transform(
            MLPQFunction,
            (hidden_sizes, activation),
            (sample_state, sample_action),
            rng
        )
        self.q2, self.q2_params = self._init_hk_transform(
            MLPQFunction,
            (hidden_sizes, activation),
            (sample_state, sample_action),
            rng
        )

    def _init_hk_transform(self, module: type, args: tuple, sample_input: tuple, rng: jnp.ndarray) -> tuple[hk.Transformed, hk.Params]:
        transform = hk.transform(lambda x: module(*args)(*x if type(x) == tuple else x))
        params = transform.init(rng, sample_input)
        return transform, params

    def __call__(self, obs):
        return self.pi.apply(obs)