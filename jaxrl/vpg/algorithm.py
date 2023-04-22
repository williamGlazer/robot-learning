import optax
import numpy as np
import jax
from optax import adam
import gym
import time
import jaxrl.vpg.core as core
from jaxrl.utils.logx import EpochLogger
import cProfile
import os.path as osp


# class needs to use raw numpy to avoid copying a table of immutable JNP arrays
class VPGBuffer:
    """
    A buffer for storing trajectories experienced by a VPG agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = (
            np.mean(self.adv_buf),
            np.std(self.adv_buf) if (self.adv_buf != 0).any() else 1.0,
        )
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(
            obs=self.obs_buf,
            act=self.act_buf,
            ret=self.ret_buf,
            adv=self.adv_buf,
            logp=self.logp_buf,
        )
        return data


def vpg(
    env_fn,
    actor_critic=core.MLPActorCritic,
    ac_kwargs=dict(),
    seed=0,
    steps_per_epoch=4000,
    epochs=50,
    gamma=0.99,
    pi_lr=3e-4,
    vf_lr=1e-3,
    train_v_iters=80,
    lam=0.97,
    max_ep_len=1000,
    logger_kwargs=dict(),
    save_freq=10,
):
    """
    Vanilla Policy Gradient

    (with GAE-Lambda for advantage estimation)

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v``
            module. The ``step`` method should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing
                                           | the log probability, according to
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical:
                                           | make sure to flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to VPG.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_v_iters (int): Number of gradient descent steps to take on
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    key = jax.random.PRNGKey(seed)

    # Instantiate environment
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic module
    key, ac_rng = jax.random.split(key)
    sample_state, _ = env.reset()
    ac = actor_critic(env.action_space, ac_rng, sample_state, **ac_kwargs)

    # Count variables
    var_counts = sum(
        jax.tree_leaves(jax.tree_map(lambda x: x.size, ac.pi_params))
    ), sum(jax.tree_leaves(jax.tree_map(lambda x: x.size, ac.v_params)))
    logger.log("\nNumber of parameters: \t pi: %d, \t v: %d\n" % var_counts)

    # Set up experience buffer
    buf = VPGBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)

    # Set up function for computing VPG policy loss
    @jax.jit
    def compute_loss_pi(pi_params, data, rng):
        obs, act, adv, logp_old = data["obs"], data["act"], data["adv"], data["logp"]

        # Policy loss
        pi = ac.pi.apply(pi_params, x=obs, rng=rng)
        logp = ac._log_prob_from_distribution(pi=pi, act=act)
        loss_pi = -(logp * adv).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean()
        ent = pi.entropy().mean()
        pi_info = dict(kl=approx_kl, ent=ent)

        return loss_pi, pi_info

    # Set up function for computing value loss
    @jax.jit
    def compute_loss_v(v_params, data, rng):
        obs, ret = data["obs"], data["ret"]
        return ((ac.v.apply(v_params, x=obs, rng=rng) - ret) ** 2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = adam(pi_lr)
    pi_opt_state = pi_optimizer.init(ac.pi_params)

    vf_optimizer = adam(vf_lr)
    vf_opt_state = vf_optimizer.init(ac.v_params)

    # Set up model saving
    # logger.setup_pytorch_saver(ac)
    @jax.jit
    def update_pi(pi_opt_state, pi_params, data, rng):
        # Train policy with a single step of gradient descent
        (loss_pi, pi_info), pi_gradient = jax.value_and_grad(
            compute_loss_pi, has_aux=True
        )(pi_params, data, rng)
        pi_updates, pi_opt_state = pi_optimizer.update(
            updates=pi_gradient, state=pi_opt_state, params=pi_params
        )
        pi_new_params = optax.apply_updates(params=pi_params, updates=pi_updates)
        return pi_opt_state, pi_new_params, loss_pi, pi_info

    @jax.jit
    def update_vf(vf_opt_state, v_params, data, rng):
        loss_v, v_gradient = jax.value_and_grad(compute_loss_v)(v_params, data, rng)
        vf_updates, vf_opt_state = vf_optimizer.update(
            updates=v_gradient, state=vf_opt_state, params=v_params
        )
        vf_new_params = optax.apply_updates(params=v_params, updates=vf_updates)
        return vf_opt_state, vf_new_params, loss_v

    def update(pi_opt_state, vf_opt_state, step_rng, warmup=False):
        data = buf.get()

        def update_pi_profiling(pi_opt_state, data, step_rng):
            return update_pi(pi_opt_state, ac.pi_params, data, step_rng)
        pi_opt_state, pi_params, loss_pi, pi_info = update_pi_profiling(
            pi_opt_state, data, step_rng
        )


        if not warmup:
            ac.pi_params = pi_params
        old_loss_pi, _ = compute_loss_pi(ac.pi_params, data, step_rng)

        # Value function learning
        for i in range(train_v_iters):
            vf_opt_state, vf_params, loss_v = update_vf(
                vf_opt_state, ac.v_params, data, step_rng
            )
            if not warmup:
                ac.v_params = vf_params
            if i == 0:
                first_loss_v = loss_v

        # Log changes from update
        kl, ent = pi_info["kl"], pi_info["ent"]
        logger.store(
            LossPi=loss_pi,
            LossV=first_loss_v,
            KL=kl,
            Entropy=ent,
            DeltaLossPi=old_loss_pi - loss_pi,
            DeltaLossV=loss_v - first_loss_v,
        )

        return pi_opt_state, vf_opt_state

    print("warmup...")
    update(pi_opt_state, vf_opt_state, key, warmup=True)
    print("Done")

    # Prepare for interaction with environment
    start_time = time.time()
    (o, _), ep_ret, ep_len = env.reset(), 0, 0

    profiler = cProfile.Profile()
    profiler.enable()
    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(steps_per_epoch):
            key, step_rng = jax.random.split(key)
            a, v, logp = ac.forward(ac.pi_params, ac.v_params, o, step_rng)

            next_o, r, d, _, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            # save and log
            buf.store(o, a, r, v, logp)
            logger.store(VVals=v)

            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == steps_per_epoch

            if terminal or epoch_ended:
                if epoch_ended and not (terminal):
                    print(
                        "Warning: trajectory cut off by epoch at %d steps." % ep_len,
                        flush=True,
                    )
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _ = ac.forward(ac.pi_params, ac.v_params, o, step_rng)
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                (o, _), ep_ret, ep_len = env.reset(), 0, 0

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({"env": env}, None)

        # Perform VPG update!
        pi_opt_state, vf_opt_state = update(pi_opt_state, vf_opt_state, step_rng)

        # Log info about epoch
        logger.log_tabular("Epoch", epoch)
        logger.log_tabular("EpRet", with_min_and_max=True)
        logger.log_tabular("EpLen", average_only=True)
        logger.log_tabular("VVals", with_min_and_max=True)
        logger.log_tabular("TotalEnvInteracts", (epoch + 1) * steps_per_epoch)
        logger.log_tabular("LossPi", average_only=True)
        logger.log_tabular("LossV", average_only=True)
        logger.log_tabular("DeltaLossPi", average_only=True)
        logger.log_tabular("DeltaLossV", average_only=True)
        logger.log_tabular("Entropy", average_only=True)
        logger.log_tabular("KL", average_only=True)
        logger.log_tabular("Time", time.time() - start_time)
        logger.dump_tabular()

    profiler.disable()

    from pathlib import Path
    prof_idx = 0
    dir = Path('./prof')
    dir.mkdir(exist_ok=True)
    while True:
        prof_file = f'prof/jax_vpg_{prof_idx}'
        if not osp.exists(prof_file):
            break
        prof_idx += 1
    profiler.dump_stats(prof_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="HalfCheetah-v2")
    parser.add_argument("--hid", type=int, default=64)
    parser.add_argument("--l", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--exp_name", type=str, default="vpg")
    args = parser.parse_args()

    from jaxrl.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(
        args.exp_name, seed=args.seed
    )

    vpg(
        lambda: gym.make(args.env),
        actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=(128, 64), activation=jax.nn.tanh),
        gamma=args.gamma,
        seed=args.seed,
        steps_per_epoch=args.steps,
        epochs=args.epochs,
        logger_kwargs=logger_kwargs,
    )
