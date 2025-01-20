# train.py
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tidal.core import (IOT, WaveFunction, TautochroneOperator, ObservationalDensity, 
                       DoublyLinkedCausalEvolution, parameterized_warping_function, 
                       parameterized_hamiltonian, parameterized_complexity_function, 
                       generate_basis_functions)
from tidal.traversal import IOTTraversal
from tidal.backprop import IOTAdaptiveOptimizer, TIDALOptimizer, TIDALTrainer, cosine_annealing_scheduler, ForexLoss, ForexTIDALOptimizer, AdaptiveLRScheduler, WaveAwareTIDALOptimizer
from typing import Tuple, List, Dict, Any
import concurrent.futures
from numba import jit, prange
import multiprocessing
from functools import partial
from tidal.iot_db import IOTDatabase, IOTModelLoader

class DataLoader:
    def __init__(self, num_workers: int = None):
        self.num_workers = num_workers or multiprocessing.cpu_count()

    @staticmethod
    def _parse_date(timestamp: int, reference_date: datetime) -> datetime:
        return reference_date + timedelta(minutes=int(timestamp))

    def _process_chunk(self, data: Tuple[pd.DataFrame, int]) -> pd.DataFrame:
        """Process a chunk of data with technical indicators."""
        chunk, idx = data
        # Create a copy to avoid pandas warnings
        chunk = chunk.copy()
        
        # Calculate technical indicators
        chunk.loc[:, 'Return'] = chunk['Close'].pct_change()
        chunk.loc[:, 'LogReturn'] = np.log(chunk['Close']) - np.log(chunk['Open'])
        chunk.loc[:, 'Volatility'] = chunk['Return'].rolling(window=21).std()
        chunk.loc[:, 'SMA_50'] = chunk['Close'].rolling(window=50).mean()
        chunk.loc[:, 'SMA_200'] = chunk['Close'].rolling(window=200).mean()
        
        # RSI calculation
        diff = chunk['Close'].diff()
        gain = (diff.copy().where(diff > 0, 0)).rolling(window=14).mean()
        loss = (-diff.copy().where(diff < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        chunk.loc[:, 'RSI'] = 100 - (100 / (1 + rs))
        
        # MACD calculation
        ema12 = chunk['Close'].ewm(span=12, adjust=False).mean()
        ema26 = chunk['Close'].ewm(span=26, adjust=False).mean()
        chunk.loc[:, 'MACD'] = ema12 - ema26
        chunk.loc[:, 'Signal'] = chunk['MACD'].ewm(span=9, adjust=False).mean()
        
        return chunk

    def load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess data using parallel processing."""
        with open(file_path, 'r') as file:
            data = json.load(file)
        reference_date = datetime(2000, 1, 1)
        timestamps = [self._parse_date(t, reference_date) for t in data['time']]
        df = pd.DataFrame({
            'Time': timestamps,
            'Open': data['open'],
            'High': data['high'],
            'Low': data['low'],
            'Close': data['close'],
            'Volume': data['volume']
        })
        
        df.set_index('Time', inplace=True)
        
        # Split data into chunks for parallel processing
        chunk_size = max(2000, len(df) // self.num_workers)
        chunks = [(df[i:i + chunk_size], i) for i in range(0, len(df), chunk_size)]
        
        # Process chunks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            processed_chunks = list(executor.map(self._process_chunk, chunks))
        # Combine processed chunks
        df = pd.concat(processed_chunks)
        df.sort_index(inplace=True)
        df.dropna(inplace=True)
        return df

@jit(nopython=True, parallel=True)
def _compute_means(features: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute means for feature groups using Numba."""
    n_samples = len(features) - lookback
    u = np.zeros(n_samples)
    v = np.zeros(n_samples)
    t = np.zeros(n_samples)
    
    for i in prange(n_samples):
        # Average of Open, High, Low
        u[i] = np.sum(features[i + lookback, :3]) / 3
        # Average of Volume, Return, LogReturn, Volatility
        v[i] = np.sum(features[i + lookback, 3:7]) / 4
        # Average of remaining features
        t[i] = np.sum(features[i + lookback, 7:]) / (features.shape[1] - 7)
    
    # Ensure coordinates are within [0, 2π]
    return (u % (2*np.pi), v % (2*np.pi), t % (2*np.pi))

class IOTMapper:
    def __init__(self, iot: IOT, lookback: int):
        self.iot = iot
        self.lookback = lookback
        self.feature_scaler = MinMaxScaler(feature_range=(0, 2*np.pi))
        self.target_scaler = MinMaxScaler(feature_range=(0, 2*np.pi))

    def map_to_iot(self, data: pd.DataFrame) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], 
                                                     np.ndarray, 
                                                     MinMaxScaler, 
                                                     MinMaxScaler]:
        """Map data to IOT coordinates using parallel processing."""
        features = ['Open', 'High', 'Low', 'Volume', 'Return', 'LogReturn', 
                   'Volatility', 'SMA_50', 'SMA_200', 'RSI', 'MACD', 'Signal', 'Close']
        target = 'Close'
        # Scale features and target
        scaled_features = self.feature_scaler.fit_transform(data[features].values)
        scaled_target = self.target_scaler.fit_transform(data[[target]].values)
        # Prepare sequences using Numba-optimized function
        coords = _compute_means(scaled_features, self.lookback)
        y = scaled_target[self.lookback:].flatten()
        return coords, y, self.feature_scaler, self.target_scaler

class ModelTrainer:
    def __init__(self, iot: IOT, l_max: int, m_max: int, db: IOTDatabase):
        self.iot = iot
        self.l_max = l_max
        self.m_max = m_max
        self.db = db

    def setup_model(self) -> Tuple[WaveFunction, DoublyLinkedCausalEvolution]:
        """Set up TIDAL model components."""
        basis_functions = generate_basis_functions(self.l_max, self.m_max)
        wave_function = WaveFunction(self.iot, basis_functions)
        tautochrone_op = TautochroneOperator(self.iot)
        obs_density = ObservationalDensity(
            parameterized_complexity_function(128164, 128164, 128164)
        )
        warping_func = parameterized_warping_function(1e-33, 128164, 128164)
        hamiltonian = parameterized_hamiltonian(warping_func)
        evolution = DoublyLinkedCausalEvolution(
            hamiltonian, tautochrone_op, tautochrone_op, obs_density, 
            137.035999, 1e-33, 7.2973525643e-3
        )
        
        return wave_function, evolution

    def train(self, wave_function: WaveFunction, 
             evolution: DoublyLinkedCausalEvolution,
             X_train: List[Tuple[float, float, float]], 
             y_train: np.ndarray,
             num_epochs: int,
             learning_rate: float,
             batch_size: int = 128164) -> Tuple[WaveFunction, Dict[str, float]]:
        """Train the model with wave-aware optimization."""
        # Initialize wave-aware optimizer
        optimizer = WaveAwareTIDALOptimizer(
            wave_function,
            learning_rate=learning_rate,
            initial_weights=(0.4, 0.3, 0.3),  # directional, momentum, volatility
            momentum_window=5,
            volatility_window=21,
            adaptation_rate=0.01,
            clip_value=1.0
        )
        
        # Initialize wave-aware trainer
        trainer = ModelTrainer(
            wave_function, 
            optimizer, 
            evolution, 
            batch_size=batch_size
        )
        
        # Setup learning rate scheduler
        lr_scheduler = lambda epoch, lr: cosine_annealing_scheduler(
            epoch, lr, T_max=num_epochs
        )
        
        # Train the model
        history = trainer.train(X_train, y_train, num_epochs, dt=1e-33, 
                              lr_scheduler=lr_scheduler)
        
        # Calculate final performance metrics
        final_epoch = history[-1]
        metrics = {
            'total_loss': final_epoch['directional'] + final_epoch['momentum'] + 
                         final_epoch['volatility'],
            'directional_loss': final_epoch['directional'],
            'momentum_loss': final_epoch['momentum'],
            'volatility_loss': final_epoch['volatility'],
            'final_weights': (final_epoch['alpha'], final_epoch['beta'], 
                            final_epoch['gamma'])
        }
        
        return wave_function, metrics

@jit(nopython=True)
def _predict_batch(wave_function_values: np.ndarray) -> np.ndarray:
    """JIT-optimized batch prediction."""
    return np.abs(wave_function_values)

class Predictor:
    def __init__(self, wave_function: WaveFunction, target_scaler: MinMaxScaler):
        self.wave_function = wave_function
        self.target_scaler = target_scaler

    def predict(self, X: List[Tuple[float, float, float]]) -> np.ndarray:
        """Make predictions using parallel batch processing."""
        batch_size = 128164
        batches = [X[i:i + batch_size] for i in range(0, len(X), batch_size)]
        
        def process_batch(batch):
            wave_values = np.array([self.wave_function(*coords) for coords in batch])
            return _predict_batch(wave_values)
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            raw_predictions = list(executor.map(process_batch, batches))
        
        predictions = np.concatenate(raw_predictions)
        return self.target_scaler.inverse_transform(predictions.reshape(-1, 1))[:, 0]

def main():
    # Initialize components
    data_loader = DataLoader()
    df = data_loader.load_and_preprocess_data('EURUSD_merged.json')
    
    print(f"Loaded data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Set up IOT and mapping
    iot = IOT(R=(137.035999/136)*np.pi*4, r=(.707*.707))
    mapper = IOTMapper(iot, lookback=128164)
    
    # Initialize IOT database
    db = IOTDatabase(base_path="tidal_forex_models")
    
    # Map data to IOT surface
    coords, y, feature_scaler, target_scaler = mapper.map_to_iot(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        list(zip(*coords)), y, test_size=0.01, shuffle=False
    )
    
    # Set up and train model with wave-aware optimization
    trainer = ModelTrainer(iot, l_max=1, m_max=1, db=db)
    wave_function, evolution = trainer.setup_model()
    trained_wave_function, metrics = trainer.train(
        wave_function, evolution, X_train, y_train, 
        num_epochs=500, learning_rate=1.007617639705882e-3
    )
    
    # Make predictions for evaluation
    predictor = Predictor(trained_wave_function, target_scaler)
    y_pred = predictor.predict(X_test)
    y_test_original = target_scaler.inverse_transform(y_test.reshape(-1, 1))[:, 0]
    
    # Calculate additional metrics
    mse = mean_squared_error(y_test_original, y_pred)
    mae = mean_absolute_error(y_test_original, y_pred)
    metrics.update({
        'mse': mse,
        'mae': mae
    })
    
    # Save model to database with enhanced metrics
    model_id = db.save_model(trained_wave_function, performance_metrics=metrics)
    print(f"Model saved with ID: {model_id}")
    print("\nTraining Metrics:")
    print(f"Total Loss: {metrics['total_loss']:.6f}")
    print(f"Directional Loss: {metrics['directional_loss']:.6f}")
    print(f"Momentum Loss: {metrics['momentum_loss']:.6f}")
    print(f"Volatility Loss: {metrics['volatility_loss']:.6f}")
    print(f"Final Weights (α,β,γ): {metrics['final_weights']}")
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"MAE: {metrics['mae']:.6f}")
    
    # Save results
    test_dates = df.index[-len(X_test):]
    results_df = pd.DataFrame({
        'Date': test_dates,
        'Actual': y_test_original,
        'Predicted': y_pred,
        'Absolute_Error': np.abs(y_test_original - y_pred),
        'Squared_Error': (y_test_original - y_pred)**2
    })
    
    results_df.to_csv('forex_prediction_comparison.csv', index=False)
    print(f"\nResults saved to forex_prediction_comparison.csv")