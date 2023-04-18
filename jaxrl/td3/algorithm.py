from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import gym
import time

import optax
import haiku as hk

import jaxrl.td3.core as core
from jaxrl.utils.logx import EpochLogger


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for TD3 agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return batch


def td3(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,
        polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000,
        update_after=1000, update_every=50, act_noise=0.1, target_noise=0.2,
        noise_clip=0.5, policy_delay=2, num_test_episodes=10, max_ep_len=1000,
        logger_kwargs=dict(), save_freq=1):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act``
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of
            observations as inputs, and ``q1`` and ``q2`` should accept a batch
            of observations and a batch of actions as inputs. When called,
            these should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``pi``       (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to TD3.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long
            you wait between updates, the ratio of env steps to gradient steps
            is locked to 1.

        act_noise (float): Stddev for Gaussian exploration noise added to
            policy at training time. (At test time, no noise is added.)

        target_noise (float): Stddev for smoothing noise added to target
            policy.

        noise_clip (float): Limit for absolute value of target policy
            smoothing noise.

        policy_delay (int): Policy will only be updated once every
            policy_delay times for each update of the Q-networks.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # TODO
    # torch.manual_seed(seed)
    # np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    # TODO finalize
    actor_critic = actor_critic(sample_state, sample_action, rng, env.action_space, **ac_kwargs)
    actor_critic_target = actor_critic(sample_state, sample_action, rng, env.observation_space, env.action_space, **ac_kwargs)

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [actor_critic.pi_params, actor_critic.q1_params, actor_critic.q2_params])
    logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)

    class TrainState:
        def __init__(self, ac, ac_tgt, pi_opt, q1_opt, q2_opt):
            self.ac = ac
            self.ac_tgt = ac_tgt
            self.pi_opt_state = pi_opt.init(self.ac.pi_params)
            self.q1_opt_state = q1_opt.init(self.ac.q1_params)
            self.q2_opt_state = q2_opt.init(self.ac.q2_params)

    # Set up function for computing TD3 Q-losses
    def compute_loss_q(
            q1_params,
            q2_params,
            train_state: TrainState,
            data: dict,
            rng: jnp.ndarray,
    ):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        _, q1_rng, q2_rng, pi_tgt_rng, q1_tgt_rng, q2_tgt_rng = jax.random.split(rng, num=6)

        ac = train_state.ac
        ac_tgt = train_state.ac_tgt

        q1 = ac.q1.apply(q1_params, (o, a), q1_rng)
        q2 = ac.q2.apply(q2_params, (o, a), q2_rng)

        # Bellman backup for Q functions
        pi_targ = ac_tgt.pi.apply(ac_tgt.pi_params, (o2,), pi_tgt_rng)

        # Target policy smoothing
        epsilon = jax.randn_like(pi_targ) * target_noise
        epsilon = jax.clamp(epsilon, -noise_clip, noise_clip)
        a2 = pi_targ + epsilon
        a2 = jax.clamp(a2, -act_limit, act_limit)

        # Target Q-values
        q1_pi_targ = ac_tgt.q1.apply(ac_tgt.q1_params, (o2, a2))
        q2_pi_targ = ac_tgt.q2.apply(ac_tgt.q2_params, (o2, a2))
        q_pi_targ = jax.min(q1_pi_targ, q2_pi_targ)
        backup = r + gamma * (1 - d) * q_pi_targ

        # avoid backprop through backup  (replaces `with torch.no_grad`)
        backup = jax.lax.stop_gradient(backup)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        loss_info = dict(Q1Vals=q1,
                         Q2Vals=q2)

        return loss_q, loss_info

    # Set up function for computing TD3 pi loss
    def compute_loss_pi(train_state: TrainState, data, rng):
        o = data['obs']
        ac = train_state.ac
        a = ac.pi.apply(ac.pi_params, o, rng)
        q1_pi = ac.q1(ac.q1_params, (o, a), rng)
        return -q1_pi.mean()

    # Set up optimizers for policy and q-function
    pi_optimizer = optax.adam(pi_lr)
    q1_optimizer = optax.adam(q_lr)
    q2_optimizer = optax.adam(q_lr)

    train_state = TrainState(
        ac=actor_critic,
        ac_tgt=actor_critic_target,
        pi_opt=pi_optimizer,
        q1_opt=q1_optimizer,
        q2_opt=q2_optimizer
    )

    # Set up model saving
    # logger.setup_pytorch_saver(ac)

    def update_pi(train_state: TrainState, data, rng):
        # Train policy with a single step of gradient descent
        pi_params, pi_opt_state = train_state.ac.pi_params, train_state.pi_opt_state
        loss_pi, pi_gradient = jax.value_and_grad(compute_loss_pi)(pi_params, train_state, data, rng)
        pi_updates, pi_new_opt_state = pi_optimizer.update(updates=pi_gradient, state=pi_opt_state, params=pi_params)
        pi_new_params = optax.apply_updates(params=pi_params, updates=pi_updates)
        return pi_new_opt_state, pi_new_params, loss_pi

    def update_q(train_state: TrainState, data, rng):
        # Train policy with a single step of gradient descent
        q1_params, q2_params = train_state.ac.q1_params, train_state.ac.q2_params
        q1_opt_state, q2_opt_state = train_state.q1_opt_state, train_state.q2_opt_state

        (loss, loss_info), gradient = jax.value_and_grad(compute_loss_pi, argnums=(0, 1))(q1_params, q2_params, train_state, data, rng)

        q1_updates, q1_new_opt_state = pi_optimizer.update(updates=gradient['q1_params'], state=q1_opt_state, params=q1_params)
        q1_new_params = optax.apply_updates(params=q1_params, updates=q1_updates)

        q2_updates, q2_new_opt_state = pi_optimizer.update(updates=gradient['q2_params'], state=q2_opt_state, params=q2_params)
        q2_new_params = optax.apply_updates(params=q2_params, updates=q2_updates)

        return q1_new_opt_state, q2_new_opt_state, q1_new_params, q2_new_params, loss, loss_info

    # TODO jax
    def update(train_state, data, timer, rng):
        # First run one gradient descent step for Q1 and Q2
        train_state.q1_opt_state, train_state.q1_opt_state, train_state.ac.q1_params, train_state.ac.q2_params, loss_q, loss_info = update_q(train_state, data, rng)

        # Record things
        logger.store(LossQ=loss_q, **loss_info)

        # Possibly update pi and target networks
        if timer % policy_delay == 0:

            # TODO if slow look at disabling gradient of QNets
            # Next run one gradient descent step for pi.
            train_state.pi_opt_state, train_state.ac.pi_params, loss_pi = update_pi(train_state, data, rng)

            # Record things
            logger.store(LossPi=loss_pi.item())

            # Finally, update target networks by polyak averaging.
            train_state.ac_tgt.pi_params = optax.incremental_update(
                new_tensors=train_state.ac.pi_params,
                old_tensors=train_state.ac_tgt.pi_params,
                step_size=polyak
            )
            train_state.ac_tgt.q1_params = optax.incremental_update(
                new_tensors=train_state.ac.q1_params,
                old_tensors=train_state.ac_tgt.q1_params,
                step_size=polyak
            )
            train_state.ac_tgt.q2_params = optax.incremental_update(
                new_tensors=train_state.ac.q2_params,
                old_tensors=train_state.ac_tgt.q2_params,
                step_size=polyak
            )

    def get_action(o, noise_scale, rng):
        a = actor_critic.act(o, rng)
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def test_agent(key):
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                key, step_rng = jax.random.split(key)
                o, r, d, _ = test_env.step(get_action(o, 0, step_rng))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        key, step_rng = jax.random.split(key)

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy (with some noise, via act_noise).
        if t > start_steps:
            a = get_action(o, act_noise, step_rng)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch, timer=j)

        # End of epoch handling
        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent(step_rng)

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='td3')
    args = parser.parse_args()

    from jaxrl.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, '../../data/')

    td3(lambda: gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
