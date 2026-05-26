# generateBurgersData.py
import numpy as np
import jax.numpy as jnp
from scipy.integrate import solve_ivp


class BurgersEquation:
    """
    1D Burgers' equation: ∂u/∂t + u*∂u/∂x = ν*∂²u/∂x²
    """

    def __init__(self, nu, x_domain=[0, 1], n_spatial=100):
        self.nu = nu  # Viscosity
        self.x_domain = x_domain
        self.n_spatial = n_spatial
        self.dx = (x_domain[1] - x_domain[0]) / (n_spatial - 1)
        self.x_grid = np.linspace(x_domain[0], x_domain[1], n_spatial)

    def initial_condition(self, x):
        """
        Initial condition: smooth wave
        """
        # Sinusoidal wave
        return np.sin(2 * np.pi * x / (self.x_domain[1] - self.x_domain[0]))

        # Alternative: shock-like initial condition
        # return np.where(x < 0.5, 1.0, 0.0)

    def spatial_derivatives(self, u):
        """
        Compute ∂u/∂x and ∂²u/∂x² using finite differences
        Periodic boundary conditions
        """
        # First derivative (central difference)
        du_dx = np.zeros_like(u)
        du_dx[1:-1] = (u[2:] - u[:-2]) / (2 * self.dx)
        # Periodic boundaries
        du_dx[0] = (u[1] - u[-2]) / (2 * self.dx)
        du_dx[-1] = (u[1] - u[-2]) / (2 * self.dx)

        # Second derivative (central difference)
        d2u_dx2 = np.zeros_like(u)
        d2u_dx2[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / (self.dx ** 2)
        # Periodic boundaries
        d2u_dx2[0] = (u[1] - 2 * u[0] + u[-2]) / (self.dx ** 2)
        d2u_dx2[-1] = (u[1] - 2 * u[-1] + u[-2]) / (self.dx ** 2)

        return du_dx, d2u_dx2

    def rhs(self, t, u):
        """
        Right-hand side of Burgers' equation
        du/dt = -u * du/dx + nu * d²u/dx²
        """
        du_dx, d2u_dx2 = self.spatial_derivatives(u)
        return -u * du_dx + self.nu * d2u_dx2

    def solve(self, t_span, n_time_points):
        """
        Solve Burgers' equation using scipy's solve_ivp

        Returns:
        --------
        t : array, shape (n_time_points,)
        u : array, shape (n_time_points, n_spatial)
        """
        u0 = self.initial_condition(self.x_grid)
        t_eval = np.linspace(t_span[0], t_span[1], n_time_points)

        sol = solve_ivp(
            self.rhs,
            t_span,
            u0,
            t_eval=t_eval,
            method='BDF',  # Good for stiff problems
            rtol=1e-8,
            atol=1e-10
        )

        return sol.t, sol.y.T  # Transpose so shape is (n_time, n_spatial)

    def generate_data(self, n_data_points, t_span, n_physics_points):
        """
        Generate training data (sparse) and physics collocation points

        Parameters:
        -----------
        n_data_points : int
            Number of time snapshots with observations
        t_span : [t_start, t_end]
            Time interval
        n_physics_points : int
            Number of collocation points for physics loss

        Returns:
        --------
        t_data : array, shape (n_data_points,)
            Time points where we have observations
        u_data : array, shape (n_data_points, n_spatial)
            Observed values at t_data
        t_physics : array, shape (n_physics_points,)
            Time collocation points
        """
        # Generate fine-grained solution
        t_fine, u_fine = self.solve(t_span, 500)

        # Sample sparse data points
        data_indices = np.linspace(0, len(t_fine) - 1, n_data_points, dtype=int)
        t_data = t_fine[data_indices]
        u_data = u_fine[data_indices, :]

        # Physics collocation points (uniformly spaced)
        t_physics = np.linspace(t_span[0], t_span[1], n_physics_points)

        return t_data, u_data, t_physics

    def exact_solution(self, t_eval):
        """
        Get exact solution at specified time points
        """
        u0 = self.initial_condition(self.x_grid)
        sol = solve_ivp(
            self.rhs,
            [0, t_eval[-1]],
            u0,
            t_eval=t_eval,
            method='BDF',
            rtol=1e-8,
            atol=1e-10
        )
        return sol.y.T  # shape: (len(t_eval), n_spatial)