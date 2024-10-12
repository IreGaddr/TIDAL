# train.py

import json
import numpy as np
import pandas as pd

from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tidal.core import IOT, WaveFunction, TautochroneOperator, ObservationalDensity, DoublyLinkedCausalEvolution, parameterized_warping_function, parameterized_hamiltonian, parameterized_complexity_function, generate_basis_functions, normalize_data
from tidal.traversal import IOTTraversal
from tidal.backprop import IOTAdaptiveOptimizer, TIDALOptimizer, TIDALTrainer, cosine_annealing_scheduler
import pickle

def load_and_preprocess_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    reference_date = datetime(2000, 1, 1)
    timestamps = [reference_date + timedelta(minutes=int(t)) for t in data['time']]
    
    df = pd.DataFrame({
        'Time': timestamps,
        'Open': data['open'],
        'High': data['high'],
        'Low': data['low'],
        'Close': data['close'],
        'Volume': data['volume']
    })
    
    df.set_index('Time', inplace=True)
    
    # Feature engineering
    df['Return'] = df['Close'].pct_change()
    df['LogReturn'] = np.log(df['Close']) - np.log(df['Open'])
    df['Volatility'] = df['Return'].rolling(window=21).std()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['RSI'] = 100 - 100 / (1 + df['Close'].diff().rolling(window=14).mean() / df['Close'].diff().rolling(window=14).std())
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()  
    df.dropna(inplace=True)
    
    return df

def map_to_iot(data, iot, lookback):
    features = ['Open', 'High', 'Low', 'Volume', 'Return', 'LogReturn', 'Volatility', 'SMA_50', 'SMA_200', 'RSI', 'MACD', 'Signal', 'Close']
    target = 'Close'
    
    feature_scaler = MinMaxScaler(feature_range=(0, 2*np.pi))
    target_scaler = MinMaxScaler(feature_range=(0, 2*np.pi))
    
    scaled_features = feature_scaler.fit_transform(data[features])
    scaled_target = target_scaler.fit_transform(data[[target]])
    
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(scaled_features[i-lookback:i])
        y.append(scaled_target[i])
    
    X = np.array(X)
    y = np.array(y)
    
    # Combine features for u, v, t coordinates
    u = np.mean(scaled_features[lookback:, :3], axis=1)  # Average of Open, High, Low
    v = np.mean(scaled_features[lookback:, 3:7], axis=1)  # Average of Volume, Return, LogReturn, Volatility
    t = np.mean(scaled_features[lookback:, 7:], axis=1)  # Average of SMA_50, SMA_200, RSI, MACD, Signal, Close
    
    # Ensure u, v, t are within [0, 2π]
    u = u % (2*np.pi)
    v = v % (2*np.pi)
    t = t % (2*np.pi)
    
    return (u, v, t), y.flatten(), feature_scaler, target_scaler

def setup_tidal_model(iot, l_max, m_max):
    basis_functions = generate_basis_functions(l_max, m_max)
    wave_function = WaveFunction(iot, basis_functions)
    tautochrone_op = TautochroneOperator(iot)
    obs_density = ObservationalDensity(parameterized_complexity_function(262144, 262144, 262144))
    warping_func = parameterized_warping_function(1e-33, 262144, 262144)
    hamiltonian = parameterized_hamiltonian(warping_func)
    evolution = DoublyLinkedCausalEvolution(hamiltonian, tautochrone_op, tautochrone_op, obs_density, 137.035999, 1e-33, 7.2973525643e-3)
    
    return wave_function, evolution

def train_model(wave_function, evolution, X_train, y_train, num_epochs, learning_rate):
    optimizer = TIDALOptimizer(wave_function, learning_rate, delta=0.1, clip_value=1.0)
    trainer = TIDALTrainer(wave_function, optimizer, evolution)
    
    lr_scheduler = lambda epoch, lr: cosine_annealing_scheduler(epoch, lr, T_max=num_epochs)
    
    trainer.train(X_train, y_train, num_epochs, dt=1e-33, lr_scheduler=lr_scheduler)
    
    return wave_function

def predict(wave_function, X, target_scaler):
    raw_predictions = np.array([np.abs(wave_function(*coords)) for coords in X])
    return target_scaler.inverse_transform(raw_predictions.reshape(-1, 1))[:, 0]

def predict_and_compare(wave_function, X_test, y_test, target_scaler, dates):
    # Make predictions
    raw_predictions = np.array([np.abs(wave_function(*coords)) for coords in X_test])
    
    # Denormalize predictions and actual values
    y_pred = target_scaler.inverse_transform(raw_predictions.reshape(-1, 1))[:, 0]
    y_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1))[:, 0]
    
    # Create DataFrame for comparison
    df_comparison = pd.DataFrame({
        'Date': dates,
        'Actual': y_actual,
        'Predicted': y_pred
    })
    
    # Calculate errors
    df_comparison['Absolute_Error'] = np.abs(df_comparison['Actual'] - df_comparison['Predicted'])
    df_comparison['Squared_Error'] = (df_comparison['Actual'] - df_comparison['Predicted'])**2
    
    # Save to CSV
    df_comparison.to_csv('forex_prediction_comparison.csv', index=False)
    
    return df_comparison
def save_model(wave_function, iot, feature_scaler, target_scaler, filename='tidal_model.pkl'):
    model_data = {
        'wave_function_coefficients': wave_function.coefficients,
        'iot_params': {
            'R': iot.R,
            'r': iot.r
        },
        'basis_functions': wave_function.basis_functions,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved successfully to {filename}")
if __name__ == "__main__":
    # Load and preprocess data
    df = load_and_preprocess_data('EURUSD_M30.json')
    
    # Set up IOT
    iot = IOT(R=(137.035999/136)*np.pi*4, r=0.5)
    
    # Map data to IOT surface
    lookback = 1536
    (X, y, feature_scaler, target_scaler) = map_to_iot(df, iot, lookback)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(list(zip(*X)), y, test_size=0.2, shuffle=False)

    # Set up TIDAL model
    wave_function, evolution = setup_tidal_model(iot, l_max=1, m_max=1)

    # Train model
    trained_wave_function = train_model(wave_function, evolution, X_train, y_train, num_epochs=10, learning_rate=1e-4)
    test_dates = df.index[-len(X_test):]
    df_comparison = predict_and_compare(trained_wave_function, X_test, y_test, target_scaler, test_dates)
    # Make predictions
    y_pred = predict(trained_wave_function, X_test, target_scaler)

    # No need for inverse transform on y_pred, as it's done in the predict function
    y_pred_original = y_pred
    y_test_original = target_scaler.inverse_transform(y_test.reshape(-1, 1))[:, 0]

    # Evaluate model
    mae = mean_absolute_error(y_test_original, y_pred_original)
    print(f"Mean Absolute Error: {mae}")

    save_model(trained_wave_function, iot, feature_scaler, target_scaler)

    # Print model size for verification
    import os
    print(f"Saved model size: {os.path.getsize('tidal_model.pkl') / 1024:.2f} KB")