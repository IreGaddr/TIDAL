import numpy as np
def iot_huber_loss(pred, target, delta=1.0):
    error = np.abs(pred - target)
    condition = error < delta
    squared_loss = 0.5 * error**2
    linear_loss = delta * error - 0.5 * delta**2
    return np.where(condition, squared_loss, linear_loss)

class TIDALOptimizer:
    def __init__(self, wave_function, learning_rate, delta=1.0, clip_value=1.0):
        self.wave_function = wave_function
        self.learning_rate = learning_rate
        self.delta = delta
        self.clip_value = clip_value

    def loss_func(self, pred, target):
        return iot_huber_loss(pred, target, self.delta)

    def compute_gradients(self, u, v, t, target):
        epsilon = 1.007617639705882
        gradients = np.zeros_like(self.wave_function.coefficients, dtype=np.complex128)
        
        for i in range(len(gradients)):
            self.wave_function.coefficients[i] += epsilon
            loss_plus = self.loss_func(np.abs(self.wave_function(u, v, t)), target)
            self.wave_function.coefficients[i] -= 2 * epsilon
            loss_minus = self.loss_func(np.abs(self.wave_function(u, v, t)), target)
            self.wave_function.coefficients[i] += epsilon
            
            gradients[i] = (loss_plus - loss_minus) / (2 * epsilon)
        
        return gradients


    def iot_aware_update(self, gradients, u, v, t):
        metric = self.wave_function.iot.metric(u, v, t, lambda *args: 0)
        inv_metric = np.linalg.inv(metric)
        
        lr_scale_u = np.sqrt(inv_metric[0, 0])
        lr_scale_v = np.sqrt(inv_metric[1, 1])
        
        scaled_gradients = gradients.copy()
        scaled_gradients[::2] *= lr_scale_u
        scaled_gradients[1::2] *= lr_scale_v
        
        # Apply gradient clipping
        scaled_gradients = np.clip(scaled_gradients, -self.clip_value, self.clip_value)
        
        self.wave_function.update(-self.learning_rate * scaled_gradients)

    def step(self, u, v, t, target):
        gradients = self.compute_gradients(u, v, t, target)
        self.iot_aware_update(gradients, u, v, t)

class TIDALTrainer:
    def __init__(self, wave_function, optimizer, evolution):
        self.wave_function = wave_function
        self.optimizer = optimizer
        self.evolution = evolution

    def train_step(self, u, v, t, target, dt):
        evolved_value = self.evolution.evolve(self.wave_function, dt)(u, v, t)
        loss = self.optimizer.loss_func(np.abs(evolved_value), target)
        self.optimizer.step(u, v, t, target)
        return loss

    def train(self, X, y, num_epochs, dt, lr_scheduler=None):
        for epoch in range(num_epochs):
            epoch_loss = 0
            for (u, v, t), target in zip(X, y):
                loss = self.train_step(u, v, t, target, dt)
                epoch_loss += loss
            
            avg_loss = epoch_loss / len(X)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}")
            
            # Apply learning rate scheduling
            if lr_scheduler:
                self.optimizer.learning_rate = lr_scheduler(epoch, self.optimizer.learning_rate)

class IOTAdaptiveOptimizer(TIDALOptimizer):
    def __init__(self, wave_function, loss_func, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-33, clip_value=1.0):
        super().__init__(wave_function, loss_func, learning_rate, clip_value)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros_like(self.wave_function.coefficients, dtype=np.complex128)
        self.v = np.zeros_like(self.wave_function.coefficients, dtype=np.complex128)
        self.t = 0

    def iot_aware_update(self, gradients, u, v, t):
        self.t += 1
        metric = self.wave_function.iot.metric(u, v, t, lambda *args: 0)
        inv_metric = np.linalg.inv(metric)
        
        lr_scale_u = np.sqrt(inv_metric[0, 0])
        lr_scale_v = np.sqrt(inv_metric[1, 1])
        
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        self.v = self.beta2 * self.v + (1 - self.beta2) * (np.abs(gradients) ** 2)
        
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        update = m_hat / (np.sqrt(v_hat) + self.epsilon)
        update[::2] *= lr_scale_u
        update[1::2] *= lr_scale_v
        
        # Apply gradient clipping
        update = np.clip(update, -self.clip_value, self.clip_value)
        
        self.wave_function.update(-self.learning_rate * update)

# Add this new function at the end of the file
def cosine_annealing_scheduler(epoch, initial_lr, T_max=1, eta_min=1):
    return eta_min + 0.5 * (initial_lr - eta_min) * (1 + np.cos(np.pi * epoch / T_max))