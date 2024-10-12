# tidal/core.py

import numpy as np
from scipy.special import sph_harm
from functools import partial

class IOT:
    def __init__(self, R, r):
        self.R = R
        self.r = r

    def metric(self, u, v, t, W):
        g_uu = (self.R + self.r * np.cos(v))**2 + W(u, v, t)
        g_vv = self.r**2 + W(u, v, t)
        return np.array([[g_uu, 0], [0, g_vv]])
    def christoffel_symbols(self, u, v, t):
        # Compute Christoffel symbols of the second kind
        g = self.metric(u, v, t, lambda *args: 0)
        g_inv = np.linalg.inv(g)
        
        symbols = np.zeros((2, 2, 2))
        
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    symbols[i, j, k] = 0.5 * sum(g_inv[i, l] * 
                        (self.partial_derivative(g, l, j, k, u, v, t) +
                         self.partial_derivative(g, l, k, j, u, v, t) -
                         self.partial_derivative(g, j, k, l, u, v, t))
                        for l in range(2))
        
        return symbols

    def partial_derivative(self, g, i, j, k, u, v, t):
        epsilon = 1e-33
        if k == 0:  # u derivative
            return (self.metric(u + epsilon, v, t, lambda *args: 0)[i, j] -
                    self.metric(u - epsilon, v, t, lambda *args: 0)[i, j]) / (2 * epsilon)
        else:  # v derivative
            return (self.metric(u, v + epsilon, t, lambda *args: 0)[i, j] -
                    self.metric(u, v - epsilon, t, lambda *args: 0)[i, j]) / (2 * epsilon)
class BasisFunction:
    def __init__(self, l, m):
        self.l = l
        self.m = m

    def __call__(self, u, v, t):
        return sph_harm(self.m, self.l, u, v).real * np.exp(-1j * (self.l*(self.l+1)) * t)

    def du(self, u, v, t):
        epsilon = 1e-33
        return (self(u + epsilon, v, t) - self(u - epsilon, v, t)) / (2 * epsilon)

    def dv(self, u, v, t):
        epsilon = 1e-33
        return (self(u, v + epsilon, t) - self(u, v - epsilon, t)) / (2 * epsilon)

class WaveFunction:
    def __init__(self, iot, basis_functions):
        self.iot = iot
        self.basis_functions = basis_functions
        self.coefficients = np.random.randn(len(basis_functions)) + 1j * np.random.randn(len(basis_functions))

    def __call__(self, u, v, t):
        return np.sum([c * f(u, v, t) for c, f in zip(self.coefficients, self.basis_functions)])

    def gradient(self, u, v, t):
        du = np.sum([c * f.du(u, v, t) for c, f in zip(self.coefficients, self.basis_functions)])
        dv = np.sum([c * f.dv(u, v, t) for c, f in zip(self.coefficients, self.basis_functions)])
        return du, dv

    def update(self, delta):
        self.coefficients += delta

    def second_derivative_u(self, u, v, t):
        epsilon = 1e-33
        return (self.gradient(u + epsilon, v, t)[0] - self.gradient(u - epsilon, v, t)[0]) / (2 * epsilon)

    def second_derivative_v(self, u, v, t):
        epsilon = 1e-33
        return (self.gradient(u, v + epsilon, t)[1] - self.gradient(u, v - epsilon, t)[1]) / (2 * epsilon)

    def mixed_derivative(self, u, v, t):
        epsilon = 1e-33
        return (self.gradient(u + epsilon, v + epsilon, t)[0] - self.gradient(u + epsilon, v - epsilon, t)[0] -
                self.gradient(u - epsilon, v + epsilon, t)[0] + self.gradient(u - epsilon, v - epsilon, t)[0]) / (4 * epsilon**2)

class TautochroneOperator:
    def __init__(self, iot):
        self.iot = iot

    def __call__(self, wave_function, u, v, t):
        return np.abs(wave_function(u, v, t))

class ObservationalDensity:
    def __init__(self, complexity_function):
        self.complexity_function = complexity_function

    def __call__(self, wave_function, u, v, t):
        return self.complexity_function(u, v) * np.abs(wave_function(u, v, t))**2

class DoublyLinkedCausalEvolution:
    def __init__(self, H, T_past, T_future, O, alpha, beta, gamma):
        self.H = H
        self.T_past = T_past
        self.T_future = T_future
        self.O = O
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def evolve(self, wave_function, dt):
        def dPsi_dt(psi, u, v, t):
            return (
                -1j * self.H(psi)(u, v, t)
                + self.alpha * self.T_past(psi, u, v, t)
                + self.beta * self.T_future(psi, u, v, t)
                + self.gamma * self.O(psi, u, v, t)
            )

        def rk4_step(psi, u, v, t):
            k1 = dPsi_dt(psi, u, v, t)
            k2 = dPsi_dt(psi, u, v, t + 0.5 * dt)
            k3 = dPsi_dt(psi, u, v, t + 0.5 * dt)
            k4 = dPsi_dt(psi, u, v, t + dt)
            
            new_value = psi(u, v, t) + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
            return new_value

        return partial(rk4_step, wave_function)

def parameterized_warping_function(a, b, c):
    return lambda u, v, t: a * np.sin(b * u) * np.cos(c * v) * np.exp(-0.1 * t)

def parameterized_hamiltonian(potential_func):
    def H(wave_function):
        def H_operator(u, v, t):
            du, dv = wave_function.gradient(u, v, t)
            laplacian = du + dv  # Simplified Laplacian for demonstration
            return -0.5 * laplacian + potential_func(u, v, t) * wave_function(u, v, t)
        return H_operator
    return H

def parameterized_complexity_function(a, b, c):
    return lambda u, v: 1 + a * np.sin(b * u) * np.cos(c * v)

def generate_basis_functions(l_max, m_max):
    return [BasisFunction(l, m) for l in range(l_max + 1) for m in range(-min(l, m_max), min(l, m_max) + 1)]

# Add this new function at the end of the file
def normalize_data(X):
    X_normalized = []
    for coord in X:
        mean = np.mean(coord)
        std = np.std(coord)
        X_normalized.append((coord - mean) / (std + 1e-33))
    return tuple(X_normalized)
