import timeit
import jax
import matplotlib
matplotlib.use('Tkagg')
import matplotlib.pyplot as plt
import numpy as np
import optax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import optax.tree

from generateData import Underdamped_Oscillator

# Initialize data

mu = 4
k = 400
m = 1

Osc = Underdamped_Oscillator(m,mu,k)
# Create test train split, important that training points lie on physics loss points
x_training, x_physics, y_training, v_training = Osc.generate_data(11,[0,1/3],31,1e-3)

params_data = np.hstack([m, mu, k, y_training, v_training])

# Set loss term weights
model_weight = 0.2
data_weight = 1-model_weight


# Discretized objective function
def fun(params):
    theta = params[0:2]

    x_len = Osc.number_physics_loss

    x1_cut = x_len + 2
    x1_t = params[3:x1_cut]
    x1_t_1 = params[2:x1_cut - 1]

    x1_slice = params[2:Osc.number_physics_loss + 2]

    x2_t = params[x1_cut + 1:]
    x2_t_1 = params[x1_cut:-1]

    X_k = jnp.vstack([x1_t, x2_t])
    X_k_1 = jnp.vstack([x1_t_1, x2_t_1])

    X_dot_k = Osc.hermite_simpson(X_k, theta)
    X_dot_k_1 = Osc.hermite_simpson(X_k_1, theta)

    X_mid = 0.5 * (X_k + X_k_1) + (Osc.dt_physics / 8.0) * (X_dot_k_1 - X_dot_k)

    X_dot_mid = Osc.hermite_simpson(X_mid, theta)

    error_vec = (X_k - X_k_1) - (Osc.dt_physics / 6.0) * (X_dot_k + 4.0 * X_dot_mid + X_dot_k_1)

    x_training_interp = x1_slice[0:Osc.number_training]
    error_data = jnp.mean(jnp.sum((y_training - x_training_interp) ** 2))

    return model_weight*jnp.mean(jnp.sum(error_vec ** 2)) +  data_weight*error_data

# Initial guesses for parameters
params_i = np.array([3.5, 390])

X0_CN = Osc.validate_CN([1,3.5,390],Osc.x_physics) # Get initial state guesses by forward solve of system

x0 = jnp.concatenate([params_i, X0_CN.flatten()])

# Set up LBFGS optimization
def run_opt(init_params, fun, opt, max_iter, tol):
  value_and_grad_fun = optax.value_and_grad_from_state(fun)

  def step(carry):
    params, state = carry
    value, grad = value_and_grad_fun(params, state=state)
    updates, state = opt.update(
        grad, state, params, value=value, grad=grad, value_fn=fun
    )
    params = optax.apply_updates(params, updates)
    return params, state

  def continuing_criterion(carry):
    _, state = carry
    iter_num = optax.tree.get(state, 'count')
    grad = optax.tree.get(state, 'grad')
    err = optax.tree.norm(grad)
    return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))

  init_carry = (init_params, opt.init(init_params))
  final_params, final_state = jax.lax.while_loop(
      continuing_criterion, step, init_carry
  )
  return final_params, final_state

# Dimensionality = 2 * Number of physics loss points + 2 Parameters estimated
dim = 2*Osc.number_physics_loss+2
opt = optax.lbfgs()
init_params = x0
print(
    f'Initial value: {fun(init_params):.2e} '
    f'Initial gradient norm: {optax.tree.norm(jax.grad(fun)(init_params)):.2e}'
)
t1 = timeit.default_timer()
final_params, _ = run_opt(init_params, fun, opt, max_iter=20000, tol=1e-8)
print(
    f'Final value: {fun(final_params):.2e}, '
    f'Final gradient norm: {optax.tree.norm(jax.grad(fun)(final_params)):.2e}'
)
solve_time = timeit.default_timer()-t1

print(f'Time to solve {solve_time}')

x_validate = np.linspace(0,1,500)
x_val = Osc.validate_CN([1,final_params[0],final_params[1]],x_validate) # Forward solve with estimated parameters

y_exact = Osc.exact_solution(x_validate) # Forward solve with real parameters


plt.figure(1)
plt.plot(x_validate, y_exact, label="Exact solution")
plt.scatter(Osc.x_training, y_training, color="tab:orange", label="Training data")
plt.scatter(Osc.x_physics, final_params[2:(len(x_physics)+2)], color="tab:green", label="Physics points")
plt.plot(x_validate, x_val[0, :], color="tab:red", label='CN-final-params-optax')
plt.xlabel('time')
plt.ylabel('x')
plt.legend()
plt.show()


