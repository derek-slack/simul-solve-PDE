import jax
import matplotlib
matplotlib.use('Tkagg')
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from src_old.generateData import Underdamped_Oscillator
from src.PhysicalNetwork import *

# Initialize data

mu = 4
k = 400
m = 1

Osc = Underdamped_Oscillator(m,mu,k)
# Create test train split, important that training points lie on physics loss points
x_training, x_physics, y_training, v_training = Osc.generate_data(11,[0,1/3],31,1e-3)

params_data = np.hstack([m, mu, k, y_training, v_training])
model = Network({'dt': 1/30})

model.p_weight = 1
model.d_weight = 1

model.create_states(['X','V'], [['t'],['t']],[[[0,1/3]],[[0,1/3]]],[{'Num PLP':31,'Num DLP':11},{'Num PLP':31,'Num DLP':0}],[{'Num PLP':31}])
model.create_parameters(['mu','k'])

model.set_grid()


def v_equation():

    X = model.states['X']
    V = model.states['V']
    mu = model.parameters['mu']
    k = model.parameters['k']

    m = 1.0

    return -(mu / m) * V - (k / m) * X


def x_equation():
    V = model.states['V']
    return 1*V

model.set_physics_equation('X',x_equation)
model.set_physics_equation('V',v_equation)
model.discretize()
solver = Solver(model,'l-bfgs',{'maxiter':25000})

# Initial guesses for parameters
params_i = np.array([3.5, 390])

X0_CN = Osc.validate_CN([1,3.5,390],Osc.x_physics) # Get initial state guesses by forward solve of system

x0 = jnp.concatenate([params_i, X0_CN.flatten()])
data = {'X':y_training,'V':[]}

res = model.obj_func(np.hstack([params_i, x0]), data)

model.p_weight = 1
model.d_weight = 1

final_params = solver.fit(x0,data)

print(final_params)

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



