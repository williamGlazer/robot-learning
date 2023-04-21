from spinnuprl.vpg.algorithm import vpg
import torch
import gym


"""
from grid search
    data/vpg_HalfCheetah-v2_ac\(128,\ 64\)_nsteps5000/
"""
size = (128, 64)
n_steps = 5000
env_name = 'HalfCheetah-v2'
env_fn = lambda : gym.make(env_name)

# prof = torch.profiler.profile(
#     # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
#     on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./prof/test'),
#     record_shapes=True,
#     with_stack=True)


for i in range(1) :

    exp_name = f"prof_3.8_best_vpg_{env_name}_ac{size}_nsteps{n_steps}_seed{i}"
    ac_kwargs = dict(hidden_sizes=size) #, activation=torch.nn.ReLU)
    logger_kwargs = dict(output_dir='data', exp_name=exp_name)


    vpg(
        env_fn,
        ac_kwargs=ac_kwargs,
        seed=i,
        steps_per_epoch=n_steps,
        logger_kwargs=logger_kwargs,
        save_freq=1,
        epochs=25,
        # profiler=prof
        # epochs=250
    )

