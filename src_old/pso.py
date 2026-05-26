import timeit
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib
matplotlib.use('Tkagg')
import matplotlib.pyplot as plt
import numpy as np
seed = 0
key = jax.random.key(seed)

from generateData import Underdamped_Oscillator


# Initialize data

mu = 4
k = 200
m = 1

Osc = Underdamped_Oscillator(m,mu,k)

# Create test train split, important that training points lie on physics loss points

x_training, x_physics, y_training, v_training = Osc.generate_data(11,[0,1/3],31,0)

params_data = np.hstack([m, mu, k, y_training, v_training])

# Set loss term weights
model_weight = 0.3333333
data_weight = 1-model_weight

n_generations = 512
n_population = 128


# Discretized objective function
def fun(params):
    theta = params[0:2]

    x_len = Osc.number_physics_loss

    x1_cut = x_len + 2
    x1_t = params[3:x1_cut]
    x1_t_1 = params[2:x1_cut - 1]

    x1_slice = params[2:x1_cut]

    x2_t = params[x1_cut + 1:]
    x2_t_1 = params[x1_cut:-1]

    X_k = jnp.vstack([x1_t, x2_t])
    X_k_1 = jnp.vstack([x1_t_1, x2_t_1])

    X_dot_k = Osc.hermite_simpson(X_k, theta)
    X_dot_k_1 = Osc.hermite_simpson(X_k_1, theta)

    X_mid = 0.5 * (X_k + X_k_1) + (Osc.dt_physics / 8.0) * (X_dot_k_1 - X_dot_k)

    X_dot_mid = Osc.hermite_simpson(X_mid, theta)

    error_vec = (X_k - X_k_1) - (Osc.dt_physics / 6.0) * (X_dot_k + 4.0 * X_dot_mid + X_dot_k_1)

    # x_training_interp = jax.numpy.interp(Osc.x_training, Osc.x_physics, x1_slice)
    x_training_interp = x1_slice[0:Osc.number_training]
    error_data = jnp.mean(jnp.sum((y_training - x_training_interp) ** 2))

    return model_weight*jnp.mean(jnp.sum(error_vec ** 2)) +  data_weight*error_data

# Map objective function with JAX
obj_vmap = jax.vmap(fun,in_axes=0)

params_i = np.array([4.4, k]) # initial guess for parameters
sig_i = [3,100] # Set uncertainty for parameters

X0_CN = Osc.validate_CN([1,4.4,k],x_physics)

# initialize population, make sure to bracket the solution properly
x0 = jnp.concatenate([params_i, X0_CN.flatten()])
x0_stack = []
for i in range(n_population):
    params_i_loop = [(np.random.random(1)*sig_i[0]+params_i[0])[0], (np.random.random(1)*sig_i[1]+params_i[1])[0]]
    X0_CN = Osc.validate_CN([1,params_i_loop[0],params_i_loop[1]],x_physics)
    x0_stack.append(np.hstack([params_i_loop, X0_CN.flatten()]))


fn_name = "oscillator"


key, subkey = jax.random.split(key)

from evosax.algorithms import PSO

num_generations = n_generations
population_size = n_population

# Instantiate evolution strategy

es = PSO(
    population_size=population_size,
    solution=x0,  # requires a dummy solution
)

# Use default parameters
params = es.default_params
population_init = jnp.array(x0_stack)
fitness_init = obj_vmap(population_init)
# Initialize evolution strategy
key, subkey = jax.random.split(key)
state = es.init(subkey,population_init, fitness_init, params)

metrics_log = []
t1 = timeit.default_timer()
for i in range(num_generations):
    key, subkey = jax.random.split(key)
    key_ask, key_eval, key_tell = jax.random.split(subkey, 3)

    population, state = es.ask(key_ask, state, params)

    fitness = obj_vmap(population)

    state, metrics = es.tell(key_tell, population, fitness, state, params)

    # Log metrics
    metrics_log.append(metrics)

solve_time = timeit.default_timer()-t1

print(f'Time to solve {solve_time}')

# Extract the best fitness values across generations
generations = [metrics["generation_counter"] for metrics in metrics_log]
best_fitness = [metrics["best_fitness"] for metrics in metrics_log]

plt.figure(figsize=(10, 5))
plt.plot(generations, best_fitness, label="Best Fitness", marker="o", markersize=3)

plt.title("Best fitness over generations")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

x_opt = metrics['best_solution']

x_validate = np.linspace(0,1,500)
x_val = Osc.validate_CN([1,metrics['best_solution'][0],metrics['best_solution'][1]],x_validate)
print(metrics['best_solution'][0:2])
print([mu,k])

y_exact = Osc.exact_solution(x_validate)

plt.figure(2)
plt.plot(x_validate, y_exact, label="Exact solution")
plt.scatter(Osc.x_training, y_training, color="tab:orange", label="Training data")
plt.scatter(Osc.x_physics, metrics['best_solution'][2:len(Osc.x_physics)+2],color="tab:green", label='simul')
plt.plot(x_validate, x_val[0, :], color="tab:red", label='CN-final-params')
# plt.scatter(x_data, X0_CN[0, :], color="tab:brown", label='CN-init')
plt.xlabel('time')
plt.ylabel('x')
plt.legend()
plt.show()