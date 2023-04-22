from jaxrl.td3.algorithm import td3
import gym

"""
from grid search
 data/td3_HalfCheetah-v2_ac\(256,\ 256,\ 256\)_batch100
"""
env_name = 'HalfCheetah-v2'
env_fn = lambda : gym.make(env_name)
size = (256, 256, 256)
batch_size = 100

for i in range(3):
    exp_name = f"jax_prof_td3_{env_name}_ac{size}_batch{batch_size}_seed{i}"
    ac_kwargs = dict(hidden_sizes=size) #, activation=torch.nn.ReLU)
    logger_kwargs = dict(output_dir='../data', exp_name=exp_name)

    td3(
        env_fn,
        ac_kwargs=ac_kwargs,
        seed=i,
        batch_size=batch_size,
        logger_kwargs=logger_kwargs,
        save_freq=1,
        epochs=300
    )
