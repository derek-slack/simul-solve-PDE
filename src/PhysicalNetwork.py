import jax
import jax.numpy as jnp
import warnings
from sympy.utilities.lambdify import lambdify
from collections import Counter

import sympy as sp


class Network:
    def __init__(self, options_dict):
        self.parameters = {}
        self.operators = {}
        self.states_map = None
        self._set_options(options_dict)

    def _set_options(self, options_dict):
        default = {'atol':1e-6, 'Num PLP':100, 'Num DLP':10, 'Solver':'l-bfgs'}
        self.options = default.update(options_dict)
        self.dt_physics = options_dict['dt']


    def create_states(self, name, dimensions_list, min_max, state_options=None, dim_options=None):

        if hasattr(self, 'states'):
            warnings.warn("State vectors already created, new states will be appended to the existing vector")
            print(f'State vector: {self.states}')
            states = self.states
            states_map = self.states_map
            states_ind = len(states_map)
        else:
            states = {}
            states_map = {}
            states_ind = 0
        for i, st in enumerate(name):
            states[st] = StateVariable(st, dimensions_list[i], min_max[i], state_options[i], dim_options)
            states_map.update({st: states_ind+i})

        self.states = states
        self.states_map = states_map

    def create_parameters(self, names):
        for name in names:
            self.parameters[name] = sp.Symbol(name)

    def set_grid(self):
        grid_dict = {}
        for key, state in self.states.items():
            for name, dim in state.dimensions.items():
                grid_dict.update({dim:{'n':dim.options['Num PLP'],'spacing':(dim.options['Min Max'][1]-dim.options['Min Max'][0])/dim.options['Num PLP']}})
        self.grid = grid_dict
        self._create_derivative_operators()

    def _create_derivative_operators(self):
        for dim, props in self.grid.items():
            n = props['n']
            dx = props['spacing']
            if dim.name is not 't':

                # 1st Order (Central Difference)
                D1 = jnp.zeros((n, n))
                for i in range(1, n - 1):
                    D1 = D1.at[i, i - 1].set(-1 / (2 * dx))
                    D1 = D1.at[i, i + 1].set(1 / (2 * dx))

                # Boundaries (one-sided differences)
                D1 = D1.at[0, 0].set(-3 / (2 * dx))
                D1 = D1.at[0, 1].set(4 / (2 * dx))
                D1 = D1.at[0, 2].set(-1 / (2 * dx))

                D1 = D1.at[-1, -3].set(1 / (2 * dx))
                D1 = D1.at[-1, -2].set(-4 / (2 * dx))
                D1 = D1.at[-1, -1].set(3 / (2 * dx))

                # 2nd Order (Central Difference)
                D2 = jnp.zeros((n, n))
                for i in range(1, n - 1):
                    D2 = D2.at[i, i - 1].set(1 / (dx ** 2))
                    D2 = D2.at[i, i].set(-2 / (dx ** 2))
                    D2 = D2.at[i, i + 1].set(1 / (dx ** 2))

                # 2nd Order Boundary conditions,
                D2 = D2.at[0, 0].set(-2 / (dx ** 2))
                D2 = D2.at[0, 1].set(1 / (dx ** 2))

                D2 = D2.at[-1, -2].set(1 / (dx ** 2))
                D2 = D2.at[-1, -1].set(-2 / (dx ** 2))

                self.operators[dim.name] = {1: D1, 2: D2}

                print(f"Created derivative operators for '{dim}': {n}x{n} matrices")

    def _apply_finite_difference(self, state_array, deriv_key, axes_map):
        deriv_counts = Counter(deriv_key)
        out_array = state_array

        for dim_name, order in deriv_counts.items():
            target_axis = axes_map[dim_name]
            D_matrix = self.operators[dim_name][order]

            out_array = jnp.moveaxis(out_array, target_axis, -1)

            out_array = out_array @ D_matrix.T
            out_array = jnp.moveaxis(out_array, -1, target_axis)

        return out_array

    def set_physics_equation(self, state, equation_callable):
        sym_expr = equation_callable()

        ordered_syms = sorted(list(sym_expr.free_symbols), key=lambda s: s.name)
        raw_jax_func = lambdify(ordered_syms, sym_expr, modules="jax")

        def physics_loss_wrapper(X_dict, theta):
            args_dict = {}

            for state_name, state_array in X_dict.items():
                args_dict[state_name] = state_array
            for param_name in self.parameters.keys():
                args_dict[param_name] = theta[param_name]

            for sym in ordered_syms:
                if "_d_" in sym.name:
                    base_state, deriv_key = sym.name.split("_d_")
                    state_array = args_dict[base_state]
                    axes_map = self.states[base_state].axes_map

                    args_dict[sym.name] = self._apply_finite_difference(
                        state_array, deriv_key, axes_map
                    )

            ordered_args = [args_dict[sym.name] for sym in ordered_syms]
            return raw_jax_func(*ordered_args)

        self.states[state].physics_equation = physics_loss_wrapper

    def solve_physics_all(self, X, theta):
        state_sol = {}
        for name, state in self.states.items():
            state_sol.update({name:state.physics_equation(X, theta)})
        return state_sol

    def discretize(self,method='Hermite-Simpson'):
        if method == 'Hermite-Simpson':
            def physics_loss(X, theta):

                X_mid_dict = {}
                X_k_dict = {}
                X_k_1_dict = {}
                X_dot_k_dict = {}
                X_dot_k_1_dict = {}

                X_dot = self.solve_physics_all(X, theta)

                for state in X.keys():
                    X_k = X[state][1:, ...]
                    X_k_1 = X[state][0:-1,...]

                    X_dot_k = X_dot[state][1:,...]
                    X_dot_k_1 = X_dot[state][0:-1,...]

                    X_k_dict.update({state: X_k})
                    X_k_1_dict.update({state: X_k_1})

                    X_dot_k_dict.update({state: X_dot_k})
                    X_dot_k_1_dict.update({state: X_dot_k_1})

                    X_mid = 0.5 * (X_k + X_k_1) + (self.dt_physics / 8.0) * (X_dot_k_1 - X_dot_k)

                    X_mid_dict.update({state: X_mid})
                X_dot_mid = self.solve_physics_all(X_mid_dict, theta)

                total_physics_loss = 0.0

                for state_name in X.keys():
                    # Retrieve the pieces
                    X_k = X_k_dict[state_name]
                    X_k_1 = X_k_1_dict[state_name]
                    X_dot_k = X_dot_k_dict[state_name]
                    X_dot_k_1 = X_dot_k_1_dict[state_name]

                    defect = (X_k - X_k_1) - (self.dt_physics / 6.0) * (X_dot_k + 4.0 * X_dot_mid[state_name] + X_dot_k_1)

                    total_physics_loss += jnp.sum(jnp.square(defect))

                return total_physics_loss

        else:
            raise NotImplementedError
        self.physics_error = physics_loss

    def _flat_to_vec(self, x):
        X = {}
        ind = 0
        theta = {}
        for sym in self.parameters.keys():
            theta.update({sym: x[ind]})
            ind = ind + 1

        for sym, state in self.states.items():
            n_points = state.n_points
            x_state = x[ind:ind+n_points]
            X_state = jnp.reshape(x_state, state.grid_shape)
            X.update({sym: X_state})

            ind += n_points
        return X, theta

    def data_error(self, X_dict, data_dict):

        data_loss = 0.0

        for state_name, true_data in data_dict.items():

            n_dlp = self.states[state_name].options['Num DLP']

            if n_dlp == 0:
                continue

            predicted_slice = X_dict[state_name][:n_dlp, ...]


            if predicted_slice.shape != true_data.shape:
                raise ValueError(
                    f"Shape mismatch in data_error for state '{state_name}'. "
                    f"Predicted slice has shape {predicted_slice.shape}, "
                    f"but training data has shape {true_data.shape}."
                )


            data_loss += jnp.sum(jnp.square(predicted_slice - true_data))

        return data_loss


    def obj_func(self, x, data):

        state_vecs, theta = self._flat_to_vec(x)
        physics_loss = self.physics_error(state_vecs,theta)
        data_loss = self.data_error(state_vecs,data)

        return self.p_weight*physics_loss + self.d_weight*data_loss

    def _vec_to_flat(self, X, theta):
        x_parts = []

        for param_name in self.parameters.keys():
            x_parts.append(theta[param_name])

        for state_name, state in self.states.items():
            state_array = X[state_name]
            x_parts.extend(state_array.flatten())

        return jnp.array(x_parts)



class StateVariable:
    def __init__(self, name, dimensions, min_max, state_options, dim_options):
        self.name = name

        self.axes_map = {dim:i for i, dim in enumerate(dimensions)}
        self.dimensions = self._set_dimensions(dimensions, dim_options, min_max)
        self._process_dimensions()

        self.sym = sp.Symbol(name)
        self.derivative = DerivativeAccessor(name, self.dimensions)
        self.min_max = min_max

        self._set_options(state_options)

    def _set_options(self, options_dict):
        self.options = {}
        self.options.update(options_dict)

    def _set_dimensions(self, dimensions_str, dim_options, min_max):
        dimensions = {}
        for i, name in enumerate(dimensions_str):
            dimensions.update({name: StateDimension(name,i,dim_options[i], min_max[i])})
        return dimensions

    def _process_dimensions(self):
        n = 1
        grid_shape = []
        for dim in self.dimensions.values():
            n_dim = dim.options['Num PLP']
            n = n*n_dim
            grid_shape.append(n_dim)
        self.grid_shape = tuple(grid_shape)
        self.n_points = n

    def __neg__(self): return -self.sym

    def __add__(self, other): return self.sym + other

    def __radd__(self, other): return other + self.sym

    def __sub__(self, other): return self.sym - other

    def __rsub__(self, other): return other - self.sym

    def __mul__(self, other): return self.sym * other

    def __rmul__(self, other): return other * self.sym

class DerivativeAccessor:
    def __init__(self, state_name, dimensions):
        self.state_name = state_name
        self.valid_dims = dimensions

    def __getitem__(self, key):
        for dim in key:
            if dim not in self.valid_dims:
                raise ValueError(f"Dimension '{dim}' not valid. Valid: {self.valid_dims}")
        key = "".join(sorted(key))
        return sp.Symbol(f"{self.state_name}_d_{key}")

class StateDimension:
    def __init__(self, name, axes_map, dim_options, min_max):
        self.name = name
        self.axes_map = axes_map

        self.options = {}
        self._set_options(dim_options, min_max)

    def _set_options(self, options_dict, min_max):
        self.options = {'Num PLP': 100, 'Num DLP': 10,'Min Max':min_max}
        self.options.update(options_dict)

class Solver:
    def __init__(self, network: Network,solver, solver_options={}):
        self._set_options(solver_options)
        if solver == 'l-bfgs':
            import optax
            import optax.tree

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

            self.opt = optax.lbfgs()

            self.run_opt = run_opt
        self.obj_func = network.obj_func


    def _set_options(self, options_dict):
        default_options = {'atol':1e-6, 'maxiter':10000}
        options_dict.update(default_options)
        self.max_iter = options_dict['maxiter']
        self.tol = options_dict['atol']


    def fit(self, x0, data):
        obj_func = lambda x: self.obj_func(x,data)
        final_params, _ = self.run_opt(x0, obj_func, self.opt, max_iter=self.max_iter, tol=self.tol)
        return final_params