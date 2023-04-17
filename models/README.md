# JAX primer

## Distributions

Distributions with `distrax` [here](https://github.com/deepmind/distrax)

Enables return of Categorical/Gaussian distributions for actions. This facilitates actions sample with `distrib.sample(seed=key)` and to evaluate action log_prob with `distrib.log_prob(action)`.

## Modules

JAX module have the parent `hk.Module` class which enables `torch.nn` like behaviors with `output = module(input)`

The `hk.transform` wrapper enables to keep the state of the module as well as to apply functions in a deterministic way using a state with `rng`.

To use your Haiku modules in a an object, you need to build them with the following:
```Python
# we define a sample network 
# which will behave in a functionnal way
class Network(hk.Module):
    def __init__(self, hparams):
        super().__init__()
        pass
    
    # idem to .forward() but functionnal approach
    def __call__(self):
        pass


# defines a functionnal nn.transform for input x
forward = hk.transform(lambda x: Network(**hparams)(x))

# initializes the dimensions with a sample input
# stores the weights of the module within the state attribute
params = forward.init(rng, sample_input)

# after initialization, you can use it as desired
# you need to feed the state and random_state
forward.apply(params, x=input, rng=rng)
```


## Optimization

```Python
import jax

optimizer = optax.adam(learning_rate)
# Obtain the `opt_state` that contains statistics for the optimizer.
params = {'w': jnp.ones((num_weights,))}
opt_state = optimizer.init(params)

compute_loss = lambda params, x, y: optax.l2_loss(params['w'].dot(x), y)
loss, grads = jax.value_and_grad(compute_loss, argnums=0, has_aux=False)(params, xs, ys)

updates, opt_state = optimizer.update(grads, opt_state)
params = optax.apply_updates(params, updates)
```


## Deterministic RNG

```Python
key = random.PRNGKey(0)
for step in range(n_steps):
    key, subkey = random.split(key)
    fn_with_randomness(subkey)
```