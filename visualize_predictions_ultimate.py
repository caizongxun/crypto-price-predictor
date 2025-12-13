#!/usr/bin/env python3
"""
Visualize Ultimate Model Predictions
Loads the trained Ultimate Ensemble model and visualizes predictions vs actuals.
"""

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.model_trainer_ultimate import UltimateLSTMModel, UltimateGRUModel, UltimateTransformerModel, UltimateEnsembleModel
from src.data_fetcher import fetch_crypto_data, add_technical_indicators
from sklearn.preprocessing import StandardScaler
import logging
from src.utils import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def load_ultimate_model(model_path, input_size, device):
    """Load the trained Ultimate Ensemble model"""
    try:
        # Initialize sub-models
        lstm = UltimateLSTMModel(input_size).to(device)
        gru = UltimateGRUModel(input_size).to(device)
        transformer = UltimateTransformerModel(input_size).to(device)
        
        # Initialize ensemble
        model = UltimateEnsembleModel(lstm, gru, transformer).to(device)
        
        # Load weights
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def visualize_ultimate(symbol='SOL', lookback=60, future_steps=1):
    """Visualize predictions for the Ultimate model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 1. Fetch Data
    logger.info(f"Fetching data for {symbol}...")
    df = fetch_crypto_data(symbol, days=200) 
    if df is None:
        return
        
    # 2. Add Indicators
    df = add_technical_indicators(df)
    
    # 3. Prepare Features
    feature_cols = [
        'open', 'high', 'low', 'close', 'volume',
        'rsi', 'mac', 'mac_signal', 'mac_hist',
        'bb_upper', 'bb_middle', 'bb_lower',
        'atr', 'adx', 'cci', 'stoch', 'williams'
    ]
    
    # Handle missing columns if any
    available_cols = [c for c in feature_cols if c in df.columns]
    data = df[available_cols].values
    
    # Scaling
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Create sequences
    X, y, dates = [], [], []
    close_idx = available_cols.index('close')
    
    for i in range(len(data_scaled) - lookback - future_steps + 1):
        X.append(data_scaled[i:i+lookback])
        y.append(data_scaled[i+lookback+future_steps-1, close_idx])
        dates.append(df.index[i+lookback+future_steps-1])
        
    X = torch.tensor(np.array(X), dtype=torch.float32).to(device)
    y = np.array(y)
    
    # 4. Load Model (Using standard name)
    model_path = f'models/saved_models/{symbol}_model.pth'
    
    # Fallback to ultimate name if standard not found
    if not os.path.exists(model_path):
        fallback_path = f'models/saved_models/{symbol}_ultimate_model.pth'
        if os.path.exists(fallback_path):
            logger.warning(f"Standard model not found, using fallback: {fallback_path}")
            model_path = fallback_path
        else:
            logger.error(f"Model file not found: {model_path}")
            return
        
    logger.info(f"Loading model from: {model_path}")
    input_size = X.shape[2]
    model = load_ultimate_model(model_path, input_size, device)
    
    # 5. Make Predictions
    logger.info("Making predictions...")
    with torch.no_grad():
        predictions = model(X).cpu().numpy().flatten()
        
    # 6. Calculate Metrics
    mae = np.mean(np.abs(predictions - y))
    mse = np.mean((predictions - y) ** 2)
    rmse = np.sqrt(mse)
    
    logger.info(f"Model Performance:")
    logger.info(f"MAE: {mae:.6f}")
    logger.info(f"RMSE: {rmse:.6f}")
    
    # 7. Visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Price Prediction
    plt.subplot(2, 1, 1)
    plt.plot(dates[-100:], y[-100:], label='Actual Price (Normalized)', color='blue', alpha=0.7)
    plt.plot(dates[-100:], predictions[-100:], label='Predicted Price (Normalized)', color='orange', alpha=0.7, linestyle='--')
    plt.title(f'{symbol} Price Prediction - Ultimate Model (Last 100 Hours)\nMAE: {mae:.4f}', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Error Distribution
    plt.subplot(2, 1, 2)
    errors = predictions - y
    sns.histplot(errors, kde=True, color='purple', alpha=0.6)
    plt.title('Prediction Error Distribution', fontsize=14)
    plt.xlabel('Prediction Error (Normalized)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = 'analysis_plots'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'{symbol}_prediction_analysis.png')
    plt.savefig(save_path)
    logger.info(f"Plot saved to {save_path}")
    
    # Don't show plot in headless environments
    # plt.show() 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, default='SOL')
    args = parser.parse_args()
    
    visualize_ultimate(args.symbol)
