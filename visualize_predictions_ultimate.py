#!/usr/bin/env python3
"""
Visualize Ultimate Model Predictions (Updated for 20-Feature Model)
Loads the trained Ultimate Ensemble model and visualizes predictions vs actuals.

Key Update: Now uses 20 features instead of 13 to match the new model architecture.
"""

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from src.model_trainer_ultimate import UltimateLSTMModel, UltimateGRUModel, UltimateTransformerModel, UltimateEnsembleModel
from src.data_fetcher import DataFetcher
from sklearn.preprocessing import MinMaxScaler
import logging
from src.utils import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def load_ultimate_model(model_path, input_size, device):
    """Load the trained Ultimate Ensemble model"""
    try:
        # Initialize sub-models with correct input_size
        lstm = UltimateLSTMModel(input_size).to(device)
        gru = UltimateGRUModel(input_size).to(device)
        transformer = UltimateTransformerModel(input_size).to(device)
        
        # Initialize ensemble
        model = UltimateEnsembleModel(lstm, gru, transformer).to(device)
        
        # Load weights
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        
        logger.info(f"Model loaded successfully from {model_path}")
        logger.info(f"Model input size: {input_size} features")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def visualize_ultimate(symbol='SOL', lookback=60, future_steps=1):
    """Visualize predictions for the Ultimate model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        # 1. Initialize Data Fetcher
        logger.info(f"Initializing data fetcher...")
        fetcher = DataFetcher()
        
        # 2. Fetch Data from Binance (using 10000 candles to match training data)
        trading_pair = f"{symbol}/USDT"
        logger.info(f"Fetching historical data for {trading_pair}...")
        df = fetcher.fetch_ohlcv_binance(trading_pair, timeframe='1h', limit=10000)
        
        if df is None or df.empty:
            logger.error(f"Failed to fetch data for {symbol}")
            return
        
        logger.info(f"Fetched {len(df)} candles")
        
        # 3. Add Technical Indicators (includes 20 features)
        logger.info(f"Adding technical indicators...")
        df = fetcher.add_technical_indicators(df)
        
        # 4. Prepare Features (20 features matching new model)
        logger.info(f"Preparing ML features with 20-feature set...")
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'SMA_10', 'SMA_20', 'EMA_12',
            'RSI', 'MACD', 'MACD_signal',
            'BB_upper', 'BB_lower',
            'Volume_ratio', 'Daily_return',
            'ROC_1', 'ROC_5',
            'Price_accel', 'Volume_accel',
            'Stoch_K', 'Stoch_D', 'Volatility'
        ]
        
        # Check which columns are available
        available_cols = [c for c in feature_cols if c in df.columns]
        logger.info(f"Using {len(available_cols)} features: {available_cols}")
        
        if len(available_cols) != 20:
            logger.warning(f"Expected 20 features but found {len(available_cols)}")
            logger.warning(f"Missing: {set(feature_cols) - set(available_cols)}")
        
        # Remove NaN values
        df_clean = df[available_cols].dropna()
        data = df_clean.values
        
        logger.info(f"Data shape after dropping NaN: {data.shape}")
        
        # Scaling using MinMaxScaler (matching training pipeline)
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data)
        
        # Create sequences
        X, y, dates = [], [], []
        close_idx = available_cols.index('close')
        
        for i in range(len(data_scaled) - lookback - future_steps + 1):
            X.append(data_scaled[i:i+lookback])
            y.append(data_scaled[i+lookback+future_steps-1, close_idx])
            dates.append(df_clean.index[i+lookback+future_steps-1])
        
        X = torch.tensor(np.array(X), dtype=torch.float32).to(device)
        y = np.array(y)
        dates = np.array(dates)
        
        logger.info(f"Created {len(X)} sequences for prediction")
        logger.info(f"X shape: {X.shape} (samples, lookback, features)")
        
        # 5. Load Model
        model_path = f'models/saved_models/{symbol}_model.pth'
        
        # Fallback to ultimate name if standard not found
        if not os.path.exists(model_path):
            fallback_path = f'models/saved_models/{symbol}_ultimate_model.pth'
            if os.path.exists(fallback_path):
                logger.warning(f"Standard model not found, using fallback: {fallback_path}")
                model_path = fallback_path
            else:
                logger.error(f"Model file not found: {model_path}")
                logger.error(f"Please train the model first using: python train_model_ultimate.py --symbol {symbol}")
                return
        
        logger.info(f"Loading model from: {model_path}")
        input_size = X.shape[2]  # Should be 20 or 22 depending on features available
        logger.info(f"Model input size: {input_size}")
        model = load_ultimate_model(model_path, input_size, device)
        
        # 6. Make Predictions
        logger.info("Making predictions...")
        with torch.no_grad():
            predictions = model(X).cpu().numpy().flatten()
        
        # 7. Calculate Metrics
        mae = np.mean(np.abs(predictions - y))
        mse = np.mean((predictions - y) ** 2)
        rmse = np.sqrt(mse)
        r2 = 1 - (np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2))
        
        logger.info(f"\n" + "="*60)
        logger.info(f"Model Performance Metrics for {symbol}:")
        logger.info(f"  MAE:  {mae:.6f}")
        logger.info(f"  RMSE: {rmse:.6f}")
        logger.info(f"  R²:   {r2:.6f}")
        logger.info(f"  Samples: {len(predictions)}")
        logger.info(f"  Features: {input_size}")
        logger.info("="*60)
        
        # 8. Visualization
        fig = plt.figure(figsize=(16, 12))
        
        # Plot 1: Price Prediction (Last 100 points)
        ax1 = plt.subplot(2, 2, 1)
        last_n = min(100, len(predictions))
        ax1.plot(range(last_n), y[-last_n:], label='Actual Price (Normalized)', color='blue', alpha=0.8, linewidth=2)
        ax1.plot(range(last_n), predictions[-last_n:], label='Predicted Price (Normalized)', color='orange', alpha=0.7, linestyle='--', linewidth=2)
        ax1.set_title(f'{symbol} Price Prediction - Ultimate Model (Last {last_n} Hours)\nMAE: {mae:.4f}', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Normalized Price')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Error Distribution with KDE
        ax2 = plt.subplot(2, 2, 2)
        errors = predictions - y
        
        # Histogram
        n, bins, patches = ax2.hist(errors, bins=50, color='purple', alpha=0.6, density=True)
        
        # Add KDE curve
        try:
            kde = gaussian_kde(errors)
            x_range = np.linspace(errors.min(), errors.max(), 200)
            ax2.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        except:
            logger.warning("Could not compute KDE")
        
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Zero Error')
        ax2.set_title('Prediction Error Distribution', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Prediction Error (Normalized)')
        ax2.set_ylabel('Frequency (Density)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Actual vs Predicted Scatter
        ax3 = plt.subplot(2, 2, 3)
        ax3.scatter(y, predictions, alpha=0.5, s=20, color='blue')
        min_val = min(y.min(), predictions.min())
        max_val = max(y.max(), predictions.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        ax3.set_title('Actual vs Predicted Values', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Actual (Normalized)')
        ax3.set_ylabel('Predicted (Normalized)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Prediction Error Over Time
        ax4 = plt.subplot(2, 2, 4)
        ax4.plot(errors, alpha=0.7, color='green', linewidth=1)
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax4.axhline(y=mae, color='orange', linestyle='--', alpha=0.5, label=f'Mean Error: {mae:.4f}')
        ax4.axhline(y=-mae, color='orange', linestyle='--', alpha=0.5)
        ax4.set_title('Prediction Error Over Time', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Sample Index')
        ax4.set_ylabel('Error (Normalized)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_dir = 'analysis_plots'
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f'{symbol}_prediction_analysis_ultimate.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"\nPlot saved to: {save_path}")
        
        # Show the plot
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        logger.error(f"Error during visualization: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Visualize Ultimate Model Predictions')
    parser.add_argument('--symbol', type=str, default='SOL', help='Cryptocurrency symbol (default: SOL)')
    args = parser.parse_args()
    
    logger.info(f"Starting visualization for {args.symbol}...\n")
    visualize_ultimate(args.symbol)
