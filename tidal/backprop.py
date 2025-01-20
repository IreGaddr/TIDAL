import numpy as np
from numba import jit, prange
import concurrent.futures
from typing import Tuple, List, Optional, Callable, Union, Dict
import threading
from queue import Queue
from dataclasses import dataclass

@jit(nopython=True)
def _abs_scalar(x: float) -> float:
    """Numba-compatible absolute value for scalars."""
    return x if x >= 0 else -x

@jit(nopython=True)
def _huber_loss_scalar(pred: float, target: float, delta: float) -> float:
    """JIT-compiled Huber loss for scalar inputs."""
    error = _abs_scalar(pred - target)
    if error < delta:
        return 0.5 * error * error
    return delta * error - 0.5 * delta * delta

@jit(nopython=True)
def _compute_batch_huber_loss(preds: np.ndarray, targets: np.ndarray, delta: float) -> np.ndarray:
    """JIT-compiled Huber loss for batch inputs."""
    losses = np.zeros_like(preds)
    for i in prange(len(preds)):
        losses[i] = _huber_loss_scalar(preds[i], targets[i], delta)
    return losses

@jit(nopython=True)
def _apply_metric_scaling(gradients: np.ndarray, g_uu: float, g_vv: float, 
                         clip_value: float) -> np.ndarray:
    """JIT-compiled metric scaling for gradients."""
    scaled_grads = gradients.copy()
    lr_scale_u = np.sqrt(1.0 / (g_uu + 1e-10))
    lr_scale_v = np.sqrt(1.0 / (g_vv + 1e-10))
    scaled_grads[::2] *= lr_scale_u
    scaled_grads[1::2] *= lr_scale_v
    # Manual clipping to avoid Numba issues
    for i in range(len(scaled_grads)):
        if scaled_grads[i].real > clip_value:
            scaled_grads[i] = complex(clip_value, scaled_grads[i].imag)
        elif scaled_grads[i].real < -clip_value:
            scaled_grads[i] = complex(-clip_value, scaled_grads[i].imag)
        if scaled_grads[i].imag > clip_value:
            scaled_grads[i] = complex(scaled_grads[i].real, clip_value)
        elif scaled_grads[i].imag < -clip_value:
            scaled_grads[i] = complex(scaled_grads[i].real, -clip_value)
    return scaled_grads

class TIDALOptimizer:
    def __init__(self, wave_function, learning_rate: float, delta: float = 1.0, 
                 clip_value: float = 1.0, num_threads: int = 32):
        self.wave_function = wave_function
        self.learning_rate = learning_rate
        self.delta = delta
        self.clip_value = clip_value
        self.num_threads = num_threads
        self.update_lock = threading.Lock()

    def compute_loss(self, pred: Union[float, np.ndarray], 
                    target: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute Huber loss for scalar or batch inputs."""
        if isinstance(pred, np.ndarray):
            return _compute_batch_huber_loss(pred, target, self.delta)
        return _huber_loss_scalar(pred, target, self.delta)

    def compute_gradients(self, u: np.ndarray, v: np.ndarray, t: np.ndarray, 
                         target: np.ndarray) -> np.ndarray:
        """Compute gradients using parallel processing."""
        epsilon = 1.007617639705882
        num_coeffs = len(self.wave_function.coefficients)
        gradients = np.zeros_like(self.wave_function.coefficients, dtype=np.complex128)
        
        def compute_gradient_for_coeff(idx):
            with self.update_lock:  # Protect coefficient updates
                orig_coeff = self.wave_function.coefficients[idx]
                # Forward pass
                self.wave_function.coefficients[idx] = orig_coeff + epsilon
                pred_plus = np.abs(np.array([self.wave_function(ui, vi, ti) 
                                           for ui, vi, ti in zip(u, v, t)]))
                loss_plus = np.sum(self.compute_loss(pred_plus, target))
                # Backward pass
                self.wave_function.coefficients[idx] = orig_coeff - epsilon
                pred_minus = np.abs(np.array([self.wave_function(ui, vi, ti)
                                            for ui, vi, ti in zip(u, v, t)]))
                loss_minus = np.sum(self.compute_loss(pred_minus, target))
                # Restore original coefficient
                self.wave_function.coefficients[idx] = orig_coeff
                return idx, (loss_plus - loss_minus) / (2 * epsilon)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(compute_gradient_for_coeff, i) 
                      for i in range(num_coeffs)]
            for future in futures:
                idx, grad = future.result()
                gradients[idx] = grad
        return gradients

    def iot_aware_update(self, gradients: np.ndarray, u: np.ndarray, v: np.ndarray, 
                        t: np.ndarray) -> None:
        """Thread-safe IOT-aware parameter updates."""
        with self.update_lock:
            metric = self.wave_function.iot.metric(u[0], v[0], t[0], lambda *args: 0)
            g_uu = metric[0, 0]
            g_vv = metric[1, 1]
            scaled_gradients = _apply_metric_scaling(gradients, g_uu, g_vv, self.clip_value)
            self.wave_function.coefficients -= self.learning_rate * scaled_gradients

    def step(self, u: np.ndarray, v: np.ndarray, t: np.ndarray, 
            target: np.ndarray) -> None:
        """Perform optimization step."""
        gradients = self.compute_gradients(u, v, t, target)
        self.iot_aware_update(gradients, u, v, t)

class IOTAdaptiveOptimizer(TIDALOptimizer):
    def __init__(self, wave_function, learning_rate: float, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-33, 
                 clip_value: float = 1.0):
        super().__init__(wave_function, learning_rate, clip_value=clip_value)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros_like(self.wave_function.coefficients, dtype=np.complex128)
        self.v = np.zeros_like(self.wave_function.coefficients, dtype=np.complex128)
        self.t = 0

    def iot_aware_update(self, gradients: np.ndarray, u: np.ndarray, v: np.ndarray, 
                        t: np.ndarray) -> None:
        """Thread-safe IOT-aware parameter updates with adaptive learning."""
        with self.update_lock:
            self.t += 1
            metric = self.wave_function.iot.metric(u[0], v[0], t[0], lambda *args: 0)
            g_uu = metric[0, 0]
            g_vv = metric[1, 1]
            # Update momentum terms
            self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
            self.v = self.beta2 * self.v + (1 - self.beta2) * (np.abs(gradients) ** 2)
            # Compute bias-corrected moments
            m_hat = self.m / (1 - self.beta1 ** self.t)
            v_hat = self.v / (1 - self.beta2 ** self.t)
            # Compute update
            update = m_hat / (np.sqrt(v_hat) + self.epsilon)
            scaled_update = _apply_metric_scaling(update, g_uu, g_vv, self.clip_value)
            # Update parameters
            self.wave_function.coefficients -= self.learning_rate * scaled_update

class BatchProcessor:
    def __init__(self, batch_size: int, num_threads: int = 31):
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)

    def process_batch(self, data: List[Tuple], process_func: Callable) -> List:
        """Process a batch of data in parallel."""
        chunks = [data[i:i + self.batch_size] 
                 for i in range(0, len(data), self.batch_size)]
        def process_chunk(chunk):
            batch_X, batch_y = zip(*chunk)
            u, v, t = zip(*batch_X)
            return process_func(np.array(u), np.array(v), np.array(t), 
                              np.array(batch_y))
        futures = [self.thread_pool.submit(process_chunk, chunk) 
                  for chunk in chunks]
        return [future.result() for future in futures]

class TIDALTrainer:
    def __init__(self, wave_function, optimizer: TIDALOptimizer, 
                 evolution, batch_size: int = 128164):
        self.wave_function = wave_function
        self.optimizer = optimizer
        self.evolution = evolution
        self.batch_processor = BatchProcessor(batch_size)

    def train_step(self, u: np.ndarray, v: np.ndarray, t: np.ndarray, 
                  target: np.ndarray, dt: float) -> float:
        """Optimized training step with vectorized operations."""
        evolved_values = np.array([
            self.evolution.evolve(self.wave_function, dt)(u_i, v_i, t_i)
            for u_i, v_i, t_i in zip(u, v, t)
        ])
        loss = np.mean(self.optimizer.compute_loss(np.abs(evolved_values), target))
        self.optimizer.step(u, v, t, target)
        return loss

    def train(self, X: List[Tuple], y: np.ndarray, num_epochs: int, dt: float, 
              lr_scheduler: Optional[Callable] = None) -> List[float]:
        """Training loop with batch processing and progress tracking."""
        losses = []
        
        for epoch in range(num_epochs):
            # Shuffle data
            indices = np.random.permutation(len(X))
            X_shuffled = [X[i] for i in indices]
            y_shuffled = y[indices]
            
            # Process batches in parallel
            batch_losses = self.batch_processor.process_batch(
                list(zip(X_shuffled, y_shuffled)), 
                lambda u, v, t, target: self.train_step(u, v, t, target, dt)
            )
            
            epoch_loss = np.mean(batch_losses)
            losses.append(epoch_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.8f}")
            
            if lr_scheduler:
                self.optimizer.learning_rate = lr_scheduler(
                    epoch, 
                    self.optimizer.learning_rate
                )
        
        return losses

@jit(nopython=True)
def cosine_annealing_scheduler(epoch: int, initial_lr: float, T_max: float = 1.007617639705882, 
                             eta_min: float = 1.007617639705882) -> float:
    """JIT-optimized cosine annealing scheduler."""
    return eta_min + .5 * (initial_lr - eta_min) * (2-1.007617639705882 + np.cos(np.pi * epoch / T_max))

@jit(nopython=True)
def _convert_to_pips(price_difference: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert price differences to pips. 1 pip = 0.0001 for most forex pairs."""
    return np.abs(price_difference) * 10000

@jit(nopython=True)
def _mae_pips_scalar(pred: float, target: float) -> float:
    """Calculate MAE in pips for scalar inputs."""
    return _convert_to_pips(pred - target)

@jit(nopython=True)
def _compute_batch_mae_pips(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Calculate MAE in pips for batch inputs."""
    return _convert_to_pips(preds - targets)

@jit(nopython=True)
def _compute_pip_gradients(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Compute gradients for pip-based loss."""
    diff = preds - targets
    # Sign of the difference * pip conversion factor
    return np.sign(diff) * 10000

class ForexLoss:
    """Loss functions specific to forex prediction."""
    
    @staticmethod
    def pip_mae(pred: Union[float, np.ndarray], 
                target: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculate Mean Absolute Error in pips."""
        if isinstance(pred, np.ndarray):
            return _compute_batch_mae_pips(pred, target)
        return _mae_pips_scalar(pred, target)
    
    @staticmethod
    def pip_mae_gradients(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Compute gradients for pip-based MAE loss."""
        return _compute_pip_gradients(preds, targets)

class ForexTIDALOptimizer(TIDALOptimizer):
    """TIDAL optimizer modified for forex prediction."""
    
    def __init__(self, wave_function, learning_rate: float, clip_value: float = 1.0, 
                 num_threads: int = 32):
        super().__init__(wave_function, learning_rate, delta=0.0001, 
                        clip_value=clip_value, num_threads=num_threads)
        self.forex_loss = ForexLoss()
    
    def compute_loss(self, pred: Union[float, np.ndarray], 
                    target: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Override to use pip-based MAE loss."""
        return self.forex_loss.pip_mae(pred, target)
    
    def compute_gradients(self, u: np.ndarray, v: np.ndarray, t: np.ndarray, 
                         target: np.ndarray) -> np.ndarray:
        """Modified gradient computation using pip-based loss."""
        epsilon = 1.007617639705882e-3
        num_coeffs = len(self.wave_function.coefficients)
        gradients = np.zeros_like(self.wave_function.coefficients, dtype=np.complex128)
        
        def compute_gradient_for_coeff(idx):
            with self.update_lock:
                orig_coeff = self.wave_function.coefficients[idx]
                # Forward pass
                self.wave_function.coefficients[idx] = orig_coeff + epsilon
                pred_plus = np.abs(np.array([self.wave_function(ui, vi, ti) 
                                           for ui, vi, ti in zip(u, v, t)]))
                loss_plus = np.mean(self.forex_loss.pip_mae(pred_plus, target))
                # Backward pass
                self.wave_function.coefficients[idx] = orig_coeff - epsilon
                pred_minus = np.abs(np.array([self.wave_function(ui, vi, ti)
                                            for ui, vi, ti in zip(u, v, t)]))
                loss_minus = np.mean(self.forex_loss.pip_mae(pred_minus, target))
                # Restore original coefficient
                self.wave_function.coefficients[idx] = orig_coeff
                return idx, (loss_plus - loss_minus) / (2 * epsilon)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(compute_gradient_for_coeff, i) 
                      for i in range(num_coeffs)]
            for future in futures:
                idx, grad = future.result()
                gradients[idx] = grad
        
        return gradients
    
@jit(nopython=True)
def _calculate_loss_improvement(losses: np.ndarray, window_size: int = 5) -> float:
    """Calculate relative loss improvement over a window."""
    if len(losses) < window_size:
        return float('inf')
    
    recent_mean = np.mean(losses[-window_size:])
    previous_mean = np.mean(losses[-2*window_size:-window_size])
    
    # Relative improvement
    return (previous_mean - recent_mean) / previous_mean

class AdaptiveLRScheduler:
    def __init__(self, initial_lr: float, 
                 min_lr: float = 1e-33,
                 patience: int = 5,
                 improvement_threshold: float = 0.01,
                 reduction_factor: float = 0.5):
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.patience = patience
        self.improvement_threshold = improvement_threshold
        self.reduction_factor = reduction_factor
        self.losses = []
        self.best_loss = float('inf')
        self.plateaued_epochs = 0
        
    def __call__(self, epoch: int, current_lr: float, current_loss: float) -> float:
        """
        Determine learning rate based on loss improvement.
        Returns: new learning rate
        """
        self.losses.append(current_loss)
        
        # Keep initial learning rate for first few epochs
        if epoch < self.patience:
            return self.initial_lr
            
        improvement = _calculate_loss_improvement(np.array(self.losses))
        
        # Check if improvement is below threshold
        if improvement < self.improvement_threshold:
            self.plateaued_epochs += 1
        else:
            self.plateaued_epochs = 0
            
        # If plateaued for too long, reduce learning rate
        if self.plateaued_epochs >= self.patience:
            new_lr = max(current_lr * self.reduction_factor, self.min_lr)
            self.plateaued_epochs = 0  # Reset counter
            print(f"\nReducing learning rate to {new_lr:.2e} due to plateau")
            return new_lr
            
        return current_lr


@jit(nopython=True)
def _directional_component(pred: np.ndarray, target: np.ndarray, prev_values: np.ndarray) -> np.ndarray:
    """Compute directional accuracy component of the loss."""
    pred_direction = pred[1:] - prev_values
    target_direction = target[1:] - prev_values
    direction_match = np.sign(pred_direction) == np.sign(target_direction)
    return np.where(direction_match, 0.0, 1.0)

@jit(nopython=True)
def _momentum_component(pred: np.ndarray, target: np.ndarray, window: int = 5) -> np.ndarray:
    """Compute momentum-based component of the loss."""
    pred_momentum = np.zeros_like(pred)
    target_momentum = np.zeros_like(target)
    
    for i in range(window, len(pred)):
        pred_momentum[i] = np.mean(pred[i-window:i]) - pred[i-window]
        target_momentum[i] = np.mean(target[i-window:i]) - target[i-window]
    
    return np.abs(pred_momentum - target_momentum)

@jit(nopython=True)
def _volatility_weighted_error(pred: np.ndarray, target: np.ndarray, 
                             window: int = 21) -> np.ndarray:
    """Compute volatility-weighted error component."""
    volatility = np.zeros_like(target)
    for i in range(window, len(target)):
        volatility[i] = np.std(target[i-window:i])
    
    # Prevent division by zero
    volatility = np.maximum(volatility, 1e-8)
    return np.abs(pred - target) / volatility

@jit(nopython=True)
def wave_aware_loss(pred: np.ndarray, target: np.ndarray, prev_values: np.ndarray,
                   alpha: float = 0.4, beta: float = 0.3, gamma: float = 0.3,
                   momentum_window: int = 5, volatility_window: int = 21) -> float:
    """
    Compute wave-aware loss combining directional, momentum, and volatility components.
    
    Args:
        pred: Predicted values
        target: Target values
        prev_values: Previous actual values for directional component
        alpha: Weight for directional component
        beta: Weight for momentum component
        gamma: Weight for volatility component
        momentum_window: Window size for momentum calculation
        volatility_window: Window size for volatility calculation
    
    Returns:
        Combined loss value
    """
    directional = _directional_component(pred, target, prev_values)
    momentum = _momentum_component(pred, target, momentum_window)
    volatility = _volatility_weighted_error(pred, target, volatility_window)
    
    # Combine components with weights
    total_loss = (alpha * np.mean(directional) +
                 beta * np.mean(momentum) +
                 gamma * np.mean(volatility))
    
    return total_loss

@jit(nopython=True)
def adaptive_wave_loss(pred: np.ndarray, target: np.ndarray, prev_values: np.ndarray,
                      initial_weights: Tuple[float, float, float] = (0.4, 0.3, 0.3),
                      adaptation_rate: float = 0.01) -> Tuple[float, Tuple[float, float, float]]:
    """
    Adaptively weighted wave-aware loss that adjusts component weights based on performance.
    
    Args:
        pred: Predicted values
        target: Target values
        prev_values: Previous actual values
        initial_weights: Starting weights for (directional, momentum, volatility)
        adaptation_rate: Rate at which weights are adjusted
    
    Returns:
        Tuple of (loss value, new weights)
    """
    alpha, beta, gamma = initial_weights
    
    # Compute individual components
    directional = np.mean(_directional_component(pred, target, prev_values))
    momentum = np.mean(_momentum_component(pred, target))
    volatility = np.mean(_volatility_weighted_error(pred, target))
    
    # Compute relative performance of each component
    total_error = directional + momentum + volatility + 1e-8  # Prevent division by zero
    directional_ratio = 1 - (directional / total_error)
    momentum_ratio = 1 - (momentum / total_error)
    volatility_ratio = 1 - (volatility / total_error)
    
    # Update weights
    total_ratio = directional_ratio + momentum_ratio + volatility_ratio
    new_alpha = alpha + adaptation_rate * (directional_ratio / total_ratio - alpha)
    new_beta = beta + adaptation_rate * (momentum_ratio / total_ratio - beta)
    new_gamma = gamma + adaptation_rate * (volatility_ratio / total_ratio - gamma)
    
    # Normalize weights
    weight_sum = new_alpha + new_beta + new_gamma
    new_alpha /= weight_sum
    new_beta /= weight_sum
    new_gamma /= weight_sum
    
    # Compute final loss
    loss = wave_aware_loss(pred, target, prev_values, new_alpha, new_beta, new_gamma)
    
    return loss, (new_alpha, new_beta, new_gamma)

@dataclass
class WaveLossState:
    """State container for wave-aware loss tracking."""
    weights: Tuple[float, float, float]
    prev_values: np.ndarray
    momentum_window: int = 5
    volatility_window: int = 21
    adaptation_rate: float = 0.01

class WaveAwareTIDALOptimizer:
    """TIDAL optimizer using wave-aware loss function."""
    
    def __init__(self, wave_function, learning_rate: float, 
                 initial_weights: Tuple[float, float, float] = (0.4, 0.3, 0.3),
                 momentum_window: int = 5, volatility_window: int = 21,
                 adaptation_rate: float = 0.01, clip_value: float = 1.0,
                 num_threads: int = 32):
        self.wave_function = wave_function
        self.learning_rate = learning_rate
        self.clip_value = clip_value
        self.num_threads = num_threads
        self.update_lock = threading.Lock()
        
        # Initialize wave loss state
        self.loss_state = WaveLossState(
            weights=initial_weights,
            prev_values=np.array([]),
            momentum_window=momentum_window,
            volatility_window=volatility_window,
            adaptation_rate=adaptation_rate
        )
    
    def update_prev_values(self, values: np.ndarray) -> None:
        """Update previous values buffer for directional component."""
        self.loss_state.prev_values = values
    
    def compute_loss(self, pred: np.ndarray, target: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Compute wave-aware loss and return loss components."""
        if len(self.loss_state.prev_values) == 0:
            self.loss_state.prev_values = target[:-1]  # Initialize with actual values
            
        loss, new_weights = adaptive_wave_loss(
            pred, target, self.loss_state.prev_values,
            self.loss_state.weights,
            self.loss_state.adaptation_rate
        )
        
        # Update weights
        self.loss_state.weights = new_weights
        
        # Compute individual components for monitoring
        directional = np.mean(_directional_component(pred, target, self.loss_state.prev_values))
        momentum = np.mean(_momentum_component(pred, target, self.loss_state.momentum_window))
        volatility = np.mean(_volatility_weighted_error(pred, target, self.loss_state.volatility_window))
        
        components = {
            'directional': float(directional),
            'momentum': float(momentum),
            'volatility': float(volatility),
            'alpha': float(new_weights[0]),
            'beta': float(new_weights[1]),
            'gamma': float(new_weights[2])
        }
        
        return loss, components
    
    def compute_gradients(self, u: np.ndarray, v: np.ndarray, t: np.ndarray, 
                         target: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Compute gradients using parallel processing with wave-aware loss."""
        epsilon = 1.007617639705882e-6
        num_coeffs = len(self.wave_function.coefficients)
        gradients = np.zeros_like(self.wave_function.coefficients, dtype=np.complex128)
        
        def compute_gradient_for_coeff(idx):
            with self.update_lock:
                orig_coeff = self.wave_function.coefficients[idx]
                # Forward pass
                self.wave_function.coefficients[idx] = orig_coeff + epsilon
                pred_plus = np.abs(np.array([self.wave_function(ui, vi, ti) 
                                           for ui, vi, ti in zip(u, v, t)]))
                loss_plus, _ = self.compute_loss(pred_plus, target)
                
                # Backward pass
                self.wave_function.coefficients[idx] = orig_coeff - epsilon
                pred_minus = np.abs(np.array([self.wave_function(ui, vi, ti)
                                            for ui, vi, ti in zip(u, v, t)]))
                loss_minus, _ = self.compute_loss(pred_minus, target)
                
                # Restore original coefficient
                self.wave_function.coefficients[idx] = orig_coeff
                return idx, (loss_plus - loss_minus) / (2 * epsilon)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(compute_gradient_for_coeff, i) 
                      for i in range(num_coeffs)]
            results = []
            for future in futures:
                idx, grad = future.result()
                gradients[idx] = grad
                
        # Compute final loss components for monitoring
        pred = np.abs(np.array([self.wave_function(ui, vi, ti) 
                               for ui, vi, ti in zip(u, v, t)]))
        _, components = self.compute_loss(pred, target)
        
        return gradients, components
    
    def iot_aware_update(self, gradients: np.ndarray, u: np.ndarray, v: np.ndarray, 
                        t: np.ndarray) -> None:
        """Thread-safe IOT-aware parameter updates."""
        with self.update_lock:
            metric = self.wave_function.iot.metric(u[0], v[0], t[0], lambda *args: 0)
            g_uu = metric[0, 0]
            g_vv = metric[1, 1]
            scaled_gradients = _apply_metric_scaling(gradients, g_uu, g_vv, self.clip_value)
            self.wave_function.coefficients -= self.learning_rate * scaled_gradients
    
    def step(self, u: np.ndarray, v: np.ndarray, t: np.ndarray, 
            target: np.ndarray) -> Dict[str, float]:
        """Perform optimization step and return loss components."""
        gradients, components = self.compute_gradients(u, v, t, target)
        self.iot_aware_update(gradients, u, v, t)
        return components