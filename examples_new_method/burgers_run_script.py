# run_burgers.py
import timeit
import jax
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from burgers_data_generation import BurgersEquation
from src.PhysicalNetwork import Network, Solver
import numpy as np

nu_true = 0.01
x_domain = [0, 1]
n_spatial = 50


n_data_points = 61
n_physics_points = 61

t_span = [0, 0.5]
t_data_span = [0, 0.5]

# 2. Generate the exact spacing
dt_physics = (t_span[1] - t_span[0]) / (n_physics_points - 1)

# Generate or slice your data
burgers = BurgersEquation(nu=nu_true, x_domain=x_domain, n_spatial=n_spatial)

# Assuming generate_data outputs uniformly spaced points based on the inputs
t_data, u_data, t_physics = burgers.generate_data(
    n_data_points,
    t_data_span,     # Constrain data generation to the first half
    n_physics_points # Generate physics grid for the full span
)
model = Network({'dt': dt_physics})
model.p_weight = 1
model.d_weight = 1
model.create_states(
    ['U'],  # State names
    [['t', 'x']],  # Dimensions for each state
    [[t_span,[0,1]]],
    [{'Num PLP': n_physics_points,  # Physics Loss Points in time
      'Num DLP': n_data_points}],  # Data Loss Points in time
    [{'Num PLP': n_physics_points},  # Time dimension grid
     {'Num PLP': n_spatial}]  # Spatial dimension grid
)

# Create parameter to estimate
model.create_parameters(['nu'])

# Set up spatial grid
model.set_grid()


def burgers_equation():
    U = model.states['U']
    nu = model.parameters['nu']

    du_dx = U.derivative['x']
    d2u_dx2 = U.derivative['xx']

    return -U * du_dx + nu * d2u_dx2


model.set_physics_equation('U', burgers_equation)
model.discretize()

nu_init = 0.015

# Initial state guess: use the initial condition propagated in time
u0_spatial = burgers.initial_condition(burgers.x_grid)
U_init = np.tile(u0_spatial, (n_physics_points, 1))

theta_init = {'nu': nu_init}
X_init = {'U': U_init}

x0 = model._vec_to_flat(X_init, theta_init)

data = {'U': u_data}  # shape: (11, 50)


initial_loss = model.obj_func(x0, data)
print(f"  Initial loss: {initial_loss:.2e}")


solver = Solver(model, 'l-bfgs', {'maxiter': 50000, 'atol': 1e-8})

t_start = timeit.default_timer()
final_params = solver.fit(x0, data)
solve_time = timeit.default_timer() - t_start
print(f'Time to solve: {solve_time}')
X_final, theta_final = model._flat_to_vec(final_params)


print(f"Estimated viscosity (nu): {theta_final['nu']:.6f}")
print(f"True viscosity:           {nu_true:.6f}")
print(f"Relative error:           {abs(theta_final['nu'] - nu_true) / nu_true * 100:.2f}%")

u_exact = burgers.exact_solution(t_physics)

# Create figure with subplots
fig = plt.figure(figsize=(16, 10))

# Plot 1: Solution at different times
ax1 = plt.subplot(2, 3, 1)
time_snapshots = [0, n_physics_points // 4, n_physics_points // 2, 3 * n_physics_points // 4, -1]
colors = plt.cm.viridis(np.linspace(0, 1, len(time_snapshots)))

for i, t_idx in enumerate(time_snapshots):
    ax1.plot(burgers.x_grid, u_exact[t_idx, :],
             color=colors[i], linestyle='-', linewidth=2,
             label=f't={t_physics[t_idx]:.2f} (exact)')
    ax1.plot(burgers.x_grid, X_final['U'][t_idx, :],
             color=colors[i], linestyle='--', linewidth=2, alpha=0.7)

ax1.set_xlabel('x')
ax1.set_ylabel('u(x,t)')
ax1.set_title('Solution Profiles (solid=exact, dashed=estimated)')
ax1.legend()
ax1.grid(alpha=0.3)

# Plot 2: Spacetime heatmap - Exact solution
ax2 = plt.subplot(2, 3, 2)
im1 = ax2.contourf(burgers.x_grid, t_physics, u_exact, levels=20, cmap='RdBu_r')
ax2.set_xlabel('x')
ax2.set_ylabel('t')
ax2.set_title('Exact Solution u(x,t)')
plt.colorbar(im1, ax=ax2)

# Plot 3: Spacetime heatmap - Estimated solution
ax3 = plt.subplot(2, 3, 3)
im2 = ax3.contourf(burgers.x_grid, t_physics, X_final['U'], levels=20, cmap='RdBu_r')
ax3.set_xlabel('x')
ax3.set_ylabel('t')
ax3.set_title(f'Estimated Solution (ν={theta_final["nu"]:.4f})')
plt.colorbar(im2, ax=ax3)

# Plot 4: Error heatmap
ax4 = plt.subplot(2, 3, 4)
error = np.abs(X_final['U'] - u_exact)
im3 = ax4.contourf(burgers.x_grid, t_physics, error, levels=20, cmap='Reds')
ax4.scatter(burgers.x_grid, [t_data[0]] * n_spatial, c='green', s=5, alpha=0.5)
for t in t_data[1:]:
    ax4.scatter(burgers.x_grid, [t] * n_spatial, c='green', s=5, alpha=0.5)
ax4.set_xlabel('x')
ax4.set_ylabel('t')
ax4.set_title('Absolute Error (green = training data)')
plt.colorbar(im3, ax=ax4)

# Plot 5: Data fit at training times
ax5 = plt.subplot(2, 3, 5)
for i in range(min(5, n_data_points)):
    ax5.plot(burgers.x_grid, u_data[i, :], 'o-',
             label=f't={t_data[i]:.2f}', alpha=0.7)
    # Interpolate estimated solution to data times
    t_idx = np.argmin(np.abs(t_physics - t_data[i]))
    ax5.plot(burgers.x_grid, X_final['U'][t_idx, :], '--', alpha=0.7)
ax5.set_xlabel('x')
ax5.set_ylabel('u')
ax5.set_title('Data Fit (circles=data, dashed=model)')
ax5.legend()
ax5.grid(alpha=0.3)

# Plot 6: Time evolution at specific spatial point
ax6 = plt.subplot(2, 3, 6)
x_probe_idx = n_spatial // 2  # Middle of domain
ax6.plot(t_physics, u_exact[:, x_probe_idx], 'b-', linewidth=2, label='Exact')
ax6.plot(t_physics, X_final['U'][:, x_probe_idx], 'r--', linewidth=2, label='Estimated')
ax6.scatter(t_data, u_data[:, x_probe_idx], c='green', s=50,
            zorder=5, label='Training data')
ax6.set_xlabel('t')
ax6.set_ylabel('u')
ax6.set_title(f'Time Evolution at x={burgers.x_grid[x_probe_idx]:.2f}')
ax6.legend()
ax6.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('burgers_results.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# 9. Quantitative Metrics
# ============================================================================
print(f"\nQuantitative Metrics:")
print(f"  Max absolute error: {np.max(error):.2e}")
print(f"  Mean absolute error: {np.mean(error):.2e}")
print(f"  RMS error: {np.sqrt(np.mean(error ** 2)):.2e}")
print(f"  Relative L2 error: {np.linalg.norm(error) / np.linalg.norm(u_exact):.2e}")