import timeit
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Tkagg')
import matplotlib.pyplot as plt
import numpy as np
import optax
jax.config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import jax.random as jrd

import optax.tree
class Underdamped_Oscillator():
    def __init__(self,m, mu, k):
        self.m = m
        self.mu = mu
        self.k = k
        self.d = mu/2
        self.w0 = np.sqrt(k)

    def generate_data(self, number_training, min_max_training, number_physics_loss, sig=1e-8):
        self.number_training = number_training
        self.number_physics_loss = number_physics_loss

        self.dt_training = (min_max_training[1] -min_max_training[0]) / (number_training -1)
        self.dt_physics = 1 / (number_physics_loss -1)

        x_training = np.linspace(min_max_training[0], min_max_training[1], number_training)
        x_physics = np.linspace(0,1,number_physics_loss)

        y_training, v_training = self.oscillator(self.d, self.w0, x_training) + np.random.normal(0, sig, (2,number_training))

        self.x_training = x_training
        self.x_physics = x_physics

        return x_training, x_physics, y_training, v_training

    def oscillator(self, d, w0, x):
        """Defines the analytical solution to the 1D underdamped harmonic oscillator problem.
        Equations taken from: https://beltoforion.de/en/harmonic_oscillator/"""

        w = np.sqrt(w0 ** 2 - d ** 2)
        phi = np.arctan(-d / w)
        A = 1 / (2 * np.cos(phi))
        cos = np.cos(phi + w * x)
        sin = np.sin(phi + w * x)
        exp = np.exp(-d * x)
        y = exp * 2 * A * cos

        v = exp * 2 * A * (-d * cos - w * sin)

        return y, v


    def CN_setup(self,theta, dt):
        m, mu, k = theta
        A = jnp.array([[1, -dt / 2], [k * dt / 2 * m, (mu * dt / 2 * m) + 1]])
        B = jnp.array([[1, dt / 2], [-k * dt / 2 * m, 1 - (mu * dt / 2 * m)]])

        return A, B


    def validate_CN(self,params,x):
        A, B = self.CN_setup(params[0:3], (x[1] - x[0]))
        X0 = np.vstack([1, 0])
        X_vec = np.zeros([2, len(x)])
        X_vec[:, 0] = X0.flatten()
        for i in range(len(x) - 1):
            X_next = (np.matmul(np.linalg.inv(A), np.matmul(B, X0))).flatten()
            X_vec[:, i + 1] = X_next
            X0 = X_next
        return X_vec

    def hermite_simpson(self,X, theta):
        """
        X is a matrix of states of shape (2, N) where:
          Row 0 is position (x)
          Row 1 is velocity (v)
        """
        mu, k = theta

        x = X[0, :]
        v = X[1, :]

        dx = v
        dv = (-k * x - mu * v) / self.m

        return jnp.vstack([dx, dv])

    def exact_solution(self,x):
        y_training, v_training = self.oscillator(self.d, self.w0, x)
        return y_training