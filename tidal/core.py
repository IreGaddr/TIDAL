import numpy as np
from functools import partial
import concurrent.futures
from typing import List, Tuple, Callable, Any, Union
from numba import jit, prange
import numba as nb

@jit(nopython=True)
def _factorial(n: int) -> float:
    if n < 0:
        return 0.0
    if n == 0:
        return 1.0
    result = 1.0
    for i in range(1, n + 1):
        result *= i
    return result

@jit(nopython=True)
def _legendre_single(l: int, m: int, x: float) -> float:
    """Compute associated Legendre polynomial for a single point."""
    if m < 0 or m > l:
        return 0.0
    if l == 0:
        return 1.0
    if l == 1:
        if m == 0:
            return x
        if m == 1:
            return -np.sqrt(1.0 - x * x)
        return 0.0
    
    # Calculate Pmm
    pmm = 1.0
    if m > 0:
        somx2 = np.sqrt((1.0 - x) * (1.0 + x))
        fact = 1.0
        for i in range(m):
            pmm *= -fact * somx2
            fact += 2.0
    
    if l == m:
        return pmm
    
    # Calculate Pm(m+1)
    pmmp1 = x * (2 * m + 1) * pmm
    if l == m + 1:
        return pmmp1
    # Calculate Pml
    pml = 0.0
    for ll in range(m + 2, l + 1):
        pml = (x * (2 * ll - 1) * pmmp1 - (ll + m - 1) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pml
    return pml

@jit(nopython=True)
def _spherical_harmonic_single(l: int, m: int, theta: float, phi: float) -> complex:
    """Compute spherical harmonic for a single point."""
    if m < 0:
        m = -m
        phase = (-1.0)**m
    else:
        phase = 1.0
    norm = np.sqrt((2.0 * l + 1.0) * _factorial(l - m) / 
                  (4.0 * np.pi * _factorial(l + m)))
    leg = _legendre_single(l, m, np.cos(theta))
    return phase * norm * leg * np.exp(1j * m * phi)

@jit(nopython=True)
def _basis_function_value(l: int, m: int, theta: float, phi: float, t: float) -> complex:
    """Compute basis function value for a single point."""
    sph = _spherical_harmonic_single(l, m, theta, phi)
    time_factor = np.exp(-1j * (l * (l + 1)) * t)
    return sph.real * time_factor

@jit(nopython=True)
def _compute_metric_elements(R: float, r: float, u: Union[float, np.ndarray],
                           v: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute metric elements handling both scalar and array inputs."""
    cos_v = np.cos(v)
    g_uu = (R + r * cos_v)**2
    g_vv = np.full_like(g_uu, r**2)
    return g_uu, g_vv

@jit(nopython=True)
def _compute_warping(u: float, v: float, t: float, a: float, b: float, c: float) -> float:
    """JIT-compiled warping function computation."""
    return a * np.sin(b * u) * np.cos(c * v) * np.exp(-0.1 * t)

class IOT:
    def __init__(self, R: float, r: float):
        self.R = R
        self.r = r
        
    def metric(self, u: Union[float, np.ndarray], v: Union[float, np.ndarray],
               t: float, W: Callable) -> np.ndarray:
        """Compute metric tensor for scalar or array inputs."""
        u = np.asarray(u)
        v = np.asarray(v)
        is_scalar = u.ndim == 0
        g_uu, g_vv = _compute_metric_elements(self.R, self.r, u, v)
        
        if is_scalar:
            warp = W(u, v, t)
            return np.array([[g_uu + warp, 0],
                           [0, g_vv + warp]])
        else:
            warp = np.array([W(ui, vi, t) for ui, vi in zip(u, v)])
            metric = np.zeros((len(u), 2, 2))
            metric[:, 0, 0] = g_uu + warp
            metric[:, 1, 1] = g_vv + warp
            return metric

    def partial_derivative(self, g: np.ndarray, i: int, j: int, k: int,
                         u: float, v: float, t: float, epsilon: float = 1e-33) -> float:
        """Compute partial derivative of metric components."""
        if k == 0:  # u derivative
            g1 = self.metric(u + epsilon, v, t, lambda *args: 0)
            g2 = self.metric(u - epsilon, v, t, lambda *args: 0)
        else:  # v derivative
            g1 = self.metric(u, v + epsilon, t, lambda *args: 0)
            g2 = self.metric(u, v - epsilon, t, lambda *args: 0)
        if g1.ndim == 3:
            return (g1[0, i, j] - g2[0, i, j]) / (2 * epsilon)
        return (g1[i, j] - g2[i, j]) / (2 * epsilon)

    def christoffel_symbols(self, u: float, v: float, t: float) -> np.ndarray:
        """Compute Christoffel symbols."""
        g = self.metric(u, v, t, lambda *args: 0)
        if g.ndim == 3:
            g = g[0]
        g_inv = np.linalg.inv(g)
        symbols = np.zeros((2, 2, 2))
        
        def compute_symbol(indices):
            i, j, k = indices
            result = 0.0
            for l in range(2):
                result += g_inv[i, l] * (
                    self.partial_derivative(g, l, j, k, u, v, t) +
                    self.partial_derivative(g, l, k, j, u, v, t) -
                    self.partial_derivative(g, j, k, l, u, v, t)
                )
            return (i, j, k), 0.5 * result

        with concurrent.futures.ThreadPoolExecutor() as executor:
            indices = [(i, j, k) for i in range(2) for j in range(2) for k in range(2)]
            results = list(executor.map(compute_symbol, indices))

        for (i, j, k), value in results:
            symbols[i, j, k] = value
        return symbols

class BasisFunction:
    def __init__(self, l: int, m: int):
        self.l = l
        self.m = m
    def __call__(self, u: float, v: float, t: float) -> complex:
        return _basis_function_value(self.l, self.m, u, v, t)
    def du(self, u: float, v: float, t: float, epsilon: float = 1e-33) -> complex:
        return (_basis_function_value(self.l, self.m, u + epsilon, v, t) -
                _basis_function_value(self.l, self.m, u - epsilon, v, t)) / (2 * epsilon)

    def dv(self, u: float, v: float, t: float, epsilon: float = 1e-33) -> complex:
        return (_basis_function_value(self.l, self.m, u, v + epsilon, t) -
                _basis_function_value(self.l, self.m, u, v - epsilon, t)) / (2 * epsilon)

class WaveFunction:
    def __init__(self, iot: IOT, basis_functions: List[BasisFunction]):
        self.iot = iot
        self.basis_functions = basis_functions
        self.coefficients = np.random.randn(len(basis_functions)) + 1j * np.random.randn(len(basis_functions))

    def __call__(self, u: Union[float, np.ndarray], v: Union[float, np.ndarray],
                t: Union[float, np.ndarray]) -> complex:
        """Evaluate wave function. Handles both scalar and array inputs."""
        u = np.asarray(u)
        v = np.asarray(v)
        t = np.asarray(t)
        result = 0j
        for c, f in zip(self.coefficients, self.basis_functions):
            if u.ndim == 0:
                result += c * f(float(u), float(v), float(t))
            else:
                result += sum(c * f(float(ui), float(vi), float(ti))
                            for ui, vi, ti in zip(u, v, t))
        return result
    def gradient(self, u: float, v: float, t: float) -> Tuple[complex, complex]:
        du_total = 0j
        dv_total = 0j
        for c, f in zip(self.coefficients, self.basis_functions):
            du_total += c * f.du(u, v, t)
            dv_total += c * f.dv(u, v, t)
        return du_total, dv_total

    def update(self, delta: np.ndarray) -> None:
        self.coefficients += delta

class TautochroneOperator:
    def __init__(self, iot: IOT):
        self.iot = iot
    def __call__(self, wave_function: WaveFunction, u: float, v: float, t: float) -> float:
        return np.abs(wave_function(u, v, t))

class ObservationalDensity:
    def __init__(self, complexity_function: Callable):
        self.complexity_function = complexity_function

    def __call__(self, wave_function: WaveFunction, u: float, v: float, t: float) -> float:
        return self.complexity_function(u, v) * np.abs(wave_function(u, v, t))**2

class DoublyLinkedCausalEvolution:
    def __init__(self, H: Callable, T_past: TautochroneOperator, T_future: TautochroneOperator,
                 O: ObservationalDensity, alpha: float, beta: float, gamma: float):
        self.H = H
        self.T_past = T_past
        self.T_future = T_future
        self.O = O
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def evolve(self, wave_function: WaveFunction, dt: float) -> Callable:
        def dPsi_dt(psi: Any, u: float, v: float, t: float) -> complex:
            return (-1j * self.H(psi)(u, v, t) +
                    self.alpha * self.T_past(psi, u, v, t) +
                    self.beta * self.T_future(psi, u, v, t) +
                    self.gamma * self.O(psi, u, v, t))

        def rk4_step(psi: WaveFunction, u: float, v: float, t: float) -> complex:
            k1 = dPsi_dt(psi, u, v, t)
            k2 = dPsi_dt(psi, u, v, t + 0.5 * dt)
            k3 = dPsi_dt(psi, u, v, t + 0.5 * dt)
            k4 = dPsi_dt(psi, u, v, t + dt)
            return psi(u, v, t) + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        return partial(rk4_step, wave_function)

def parameterized_warping_function(a: float, b: float, c: float) -> Callable:
    return lambda u, v, t: _compute_warping(u, v, t, a, b, c)

@jit(nopython=True)
def _complexity_function(a: float, b: float, c: float, u: float, v: float) -> float:
    return 1 + a * np.sin(b * u) * np.cos(c * v)

def parameterized_complexity_function(a: float, b: float, c: float) -> Callable:
    return lambda u, v: _complexity_function(a, b, c, u, v)

def parameterized_hamiltonian(potential_func: Callable) -> Callable:
    def H(wave_function: WaveFunction) -> Callable:
        def H_operator(u: float, v: float, t: float) -> complex:
            du, dv = wave_function.gradient(u, v, t)
            laplacian = du + dv
            return -0.5 * laplacian + potential_func(u, v, t) * wave_function(u, v, t)
        return H_operator
    return H

def generate_basis_functions(l_max: int, m_max: int) -> List[BasisFunction]:
    return [BasisFunction(l, m) for l in range(l_max + 1)
            for m in range(-min(l, m_max), min(l, m_max) + 1)]

@jit(nopython=True)
def normalize_data(X: np.ndarray) -> Tuple[np.ndarray, ...]:
    """JIT-optimized data normalization."""
    X_normalized = []
    for coord in X:
        mean = np.mean(coord)
        std = np.std(coord)
        X_normalized.append((coord - mean) / (std + 1e-33))
    return tuple(X_normalized)