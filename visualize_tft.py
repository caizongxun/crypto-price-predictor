#!/usr/bin/env python3
"""
Visualize Temporal Fusion Transformer Predictions
FIX: Correctly inverse-transform scaled predictions and targets
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from src.model_tft import TemporalFusionTransformer
from src.data_fetcher_tft import TFTDataFetcher
import logging
from src.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def load_tft_model(model_path, input_size, device):
    """Load trained TFT model"""
    try:
        model = TemporalFusionTransformer(
            input_size=input_size,
            hidden_size=256,
            num_heads=8,
            num_layers=2,
            dropout=0.2
        ).to(device)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def visualize_tft(symbol='SOL', lookback=60):
    """Visualize TFT model predictions with correct scaling"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        # Initialize fetcher
        logger.info("Initializing data fetcher...")
        fetcher = TFTDataFetcher()
        
        # Fetch data
        trading_pair = f"{symbol}/USDT"
        logger.info(f"Fetching historical data for {trading_pair}...")
        df = fetcher.fetch_ohlcv_binance(trading_pair, timeframe='1h', limit=5000)
        
        if df is None or df.empty:
            logger.error(f"Failed to fetch data for {symbol}")
            return
        
        logger.info(f"Fetched {len(df)} candles")
        
        # Add indicators
        logger.info("Adding technical indicators...")
        df = fetcher.add_tft_indicators(df)
        
        # Prepare features
        logger.info("Preparing ML features...")
        X, y_original, scaler = fetcher.prepare_ml_features(df, lookback=lookback)
        
        logger.info(f"Created {len(X)} sequences for prediction")
        
        # Load model
        model_path = f'models/saved_models/{symbol}_tft_model.pth'
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            logger.error(f"Please train first: python train_tft.py --symbol {symbol}")
            return
        
        logger.info(f"Loading model from: {model_path}")
        input_size = X.shape[2]
        model = load_tft_model(model_path, input_size, device)
        
        # Make predictions (scaled)
        logger.info("Making predictions...")
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            predictions_scaled = model(X_tensor).cpu().numpy().flatten()
        
        # CRITICAL FIX: Inverse transform predictions from scaled to original price
        # The scaler was fit on the 'close' column, so we need to inverse it
        logger.info("Inverse-transforming predictions and targets...")
        
        # Reshape for inverse_transform (must be 2D)
        predictions_inverse = scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        
        # y_original is already in original scale (not normalized)
        # So we use it directly
        y = y_original
        predictions = predictions_inverse
        
        # Calculate metrics with ORIGINAL SCALE values
        mae = np.mean(np.abs(predictions - y))
        rmse = np.sqrt(np.mean((predictions - y) ** 2))
        mape = np.mean(np.abs((y - predictions) / (np.abs(y) + 1e-8))) * 100
        
        logger.info("\n" + "="*60)
        logger.info(f"Model Performance Metrics for {symbol} (Original Scale):")
        logger.info(f"  MAE:  {mae:.6f} USD")
        logger.info(f"  RMSE: {rmse:.6f} USD")
        logger.info(f"  MAPE: {mape:.4f}%")
        logger.info(f"  Samples: {len(predictions)}")
        logger.info(f"  Features: {input_size}")
        logger.info(f"  Mean Actual Price: {y.mean():.2f} USD")
        logger.info(f"  Mean Predicted Price: {predictions.mean():.2f} USD")
        logger.info("="*60)
        
        # Visualization
        fig = plt.figure(figsize=(16, 12))
        
        # Plot 1: Last 100 hours (ORIGINAL PRICES)
        ax1 = plt.subplot(2, 2, 1)
        last_n = min(100, len(predictions))
        ax1.plot(range(last_n), y[-last_n:], label='Actual Price (USD)', color='blue', alpha=0.8, linewidth=2)
        ax1.plot(range(last_n), predictions[-last_n:], label='Predicted Price (TFT, USD)', color='green', alpha=0.7, linestyle='--', linewidth=2)
        ax1.set_title(f'{symbol} Price Prediction - TFT (Last {last_n} Hours)\nMAE: {mae:.4f} USD, MAPE: {mape:.2f}%', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Price (USD)')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Error distribution
        ax2 = plt.subplot(2, 2, 2)
        errors = predictions - y
        n, bins, patches = ax2.hist(errors, bins=50, color='purple', alpha=0.6, density=True)
        
        try:
            kde = gaussian_kde(errors)
            x_range = np.linspace(errors.min(), errors.max(), 200)
            ax2.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        except:
            logger.warning("Could not compute KDE")
        
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Zero Error')
        ax2.set_title('Prediction Error Distribution (TFT)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Prediction Error (USD)')
        ax2.set_ylabel('Frequency (Density)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Actual vs predicted scatter
        ax3 = plt.subplot(2, 2, 3)
        ax3.scatter(y, predictions, alpha=0.5, s=20, color='green')
        min_val = min(y.min(), predictions.min())
        max_val = max(y.max(), predictions.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        ax3.set_title('Actual vs Predicted Values (TFT, Original Scale)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Actual (USD)')
        ax3.set_ylabel('Predicted (USD)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Error over time
        ax4 = plt.subplot(2, 2, 4)
        ax4.plot(errors, alpha=0.7, color='orange', linewidth=1)
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax4.axhline(y=mae, color='orange', linestyle='--', alpha=0.5, label=f'Mean Error: {mae:.4f} USD')
        ax4.axhline(y=-mae, color='orange', linestyle='--', alpha=0.5)
        ax4.set_title('Prediction Error Over Time (TFT)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Sample Index')
        ax4.set_ylabel('Error (USD)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_dir = 'analysis_plots'
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f'{symbol}_tft_analysis.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"\nPlot saved to: {save_path}")
        
        # Show plot
        plt.show()
        
    except Exception as e:
        logger.error(f"Error during visualization: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Visualize TFT Model Predictions')
    parser.add_argument('--symbol', type=str, default='SOL', help='Cryptocurrency symbol (default: SOL)')
    args = parser.parse_args()
    
    logger.info(f"Starting visualization for {args.symbol}...\n")
    visualize_tft(args.symbol)
