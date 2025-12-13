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
from src.data_fetcher import DataFetcher
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
        
        logger.info(f"Model loaded successfully from {model_path}")
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
        
        # 2. Fetch Data from Binance
        trading_pair = f"{symbol}/USDT"
        logger.info(f"Fetching historical data for {trading_pair}...")
        df = fetcher.fetch_ohlcv_binance(trading_pair, timeframe='1h', limit=1000)
        
        if df is None or df.empty:
            logger.error(f"Failed to fetch data for {symbol}")
            return
        
        logger.info(f"Fetched {len(df)} candles")
        
        # 3. Add Technical Indicators
        logger.info(f"Adding technical indicators...")
        df = fetcher.add_technical_indicators(df)
        
        # 4. Prepare Features
        logger.info(f"Preparing ML features...")
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'SMA_10', 'SMA_20', 'SMA_50',
            'RSI', 'MACD', 'MACD_diff',
            'BB_upper', 'BB_lower', 'ATR',
            'Volume_ratio', 'Daily_return', 'Price_momentum'
        ]
        
        # Handle missing columns if any
        available_cols = [c for c in feature_cols if c in df.columns]
        logger.info(f"Using {len(available_cols)} features: {available_cols}")
        
        # Remove NaN values
        df_clean = df[available_cols].dropna()
        data = df_clean.values
        
        # Scaling
        scaler = StandardScaler()
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
        
        # 5. Load Model (Using standard name)
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
        
        # 6. Make Predictions
        logger.info("Making predictions...")
        with torch.no_grad():
            predictions = model(X).cpu().numpy().flatten()
        
        # 7. Calculate Metrics
        mae = np.mean(np.abs(predictions - y))
        mse = np.mean((predictions - y) ** 2)
        rmse = np.sqrt(mse)
        
        logger.info(f"\n" + "="*60)
        logger.info(f"Model Performance Metrics for {symbol}:")
        logger.info(f"  MAE:  {mae:.6f}")
        logger.info(f"  RMSE: {rmse:.6f}")
        logger.info(f"  Samples: {len(predictions)}")
        logger.info("="*60)
        
        # 8. Visualization
        fig = plt.figure(figsize=(16, 10))
        
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
        
        # Plot 2: Error Distribution
        ax2 = plt.subplot(2, 2, 2)
        errors = predictions - y
        ax2.hist(errors, bins=50, kde=True, color='purple', alpha=0.6)
        ax2.axvline(x=0, color='red', linestyle='--', label='Zero Error')
        ax2.set_title('Prediction Error Distribution', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Prediction Error (Normalized)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Actual vs Predicted Scatter
        ax3 = plt.subplot(2, 2, 3)
        ax3.scatter(y, predictions, alpha=0.5, s=20)
        min_val = min(y.min(), predictions.min())
        max_val = max(y.max(), predictions.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
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
        save_path = os.path.join(output_dir, f'{symbol}_prediction_analysis.png')
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, default='SOL')
    args = parser.parse_args()
    
    logger.info(f"Starting visualization for {args.symbol}...\n")
    visualize_ultimate(args.symbol)
