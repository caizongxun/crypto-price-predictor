#!/usr/bin/env python3
"""
🚀 TFT V3 Optimized Visualization (v1.2+)

Advanced visualization with:
- Multi-step forecasting display (3-5 candles)
- Ensemble prediction smoothing
- Confidence intervals visualization
- Direction accuracy heatmaps
- Performance metrics tracking (MAE, MAPE, Direction Accuracy)
- Error analysis and volatility-adjusted metrics

Usage:
  python visualize_tft_v3_optimized.py --symbol SOL
  python visualize_tft_v3_optimized.py --symbol BTC --lookback 120 --steps 5
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from pathlib import Path
import logging
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent))

from src.utils import setup_logging
from src.data_fetcher_tft_v3 import TFTDataFetcher
from src.model_tft_v3_enhanced_optimized import TemporalFusionTransformerV3EnhancedOptimized

setup_logging()
logger = logging.getLogger(__name__)


class OptimizedEnsembleSmoother:
    """Advanced ensemble smoothing"""
    
    @staticmethod
    def kalman_smooth(data, process_var=0.001, measurement_var=0.05):
        """Kalman filter"""
        estimates = np.zeros_like(data)
        estimate = data[0]
        estimate_error = 0.5
        
        for i, measurement in enumerate(data):
            estimate_error += process_var
            kalman_gain = estimate_error / (estimate_error + measurement_var)
            estimate += kalman_gain * (measurement - estimate)
            estimate_error *= (1 - kalman_gain)
            estimates[i] = estimate
        
        return estimates
    
    @staticmethod
    def exponential_smooth(data, alpha=0.25):
        """Exponential smoothing"""
        smoothed = np.zeros_like(data)
        smoothed[0] = data[0]
        
        for i in range(1, len(data)):
            smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
        
        return smoothed
    
    @staticmethod
    def moving_average(data, window=3):
        """Moving average"""
        return uniform_filter1d(data, size=window, mode='nearest')
    
    @staticmethod
    def ensemble_smooth(data):
        """Combine multiple techniques"""
        kalman = OptimizedEnsembleSmoother.kalman_smooth(data)
        exponential = OptimizedEnsembleSmoother.exponential_smooth(kalman, alpha=0.25)
        ma = OptimizedEnsembleSmoother.moving_average(exponential, window=2)
        result = 0.6 * exponential + 0.4 * ma
        
        return result, kalman, exponential, ma


class PerformanceMetrics:
    """Compute comprehensive metrics"""
    
    @staticmethod
    def calculate(y_true, y_pred, volatility=None):
        """Calculate all metrics"""
        mae = np.mean(np.abs(y_pred - y_true))
        mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / 
                              (np.abs(y_true) + np.abs(y_pred) + 1e-8))
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        # Direction accuracy
        true_dir = np.sign(np.diff(y_true))
        pred_dir = np.sign(np.diff(y_pred))
        dir_acc = np.mean(true_dir == pred_dir) * 100
        
        # Volatility-adjusted
        mae_high_vol = None
        mae_low_vol = None
        if volatility is not None:
            high_vol_mask = volatility > np.percentile(volatility, 75)
            low_vol_mask = volatility <= np.percentile(volatility, 25)
            
            if high_vol_mask.sum() > 0:
                mae_high_vol = np.mean(np.abs(y_pred[high_vol_mask] - y_true[high_vol_mask]))
            if low_vol_mask.sum() > 0:
                mae_low_vol = np.mean(np.abs(y_pred[low_vol_mask] - y_true[low_vol_mask]))
        
        return {
            'mae': mae,
            'mape': mape,
            'rmse': rmse,
            'smape': smape,
            'r2': r2,
            'dir_acc': dir_acc,
            'mae_high_vol': mae_high_vol,
            'mae_low_vol': mae_low_vol
        }


def predict_multistep(model, X_latest, steps=5, device='cpu'):
    """Multi-step prediction"""
    predictions = []
    uncertainties = []
    
    with torch.no_grad():
        X_tensor = torch.tensor(X_latest, dtype=torch.float32).to(device)
        result = model(X_tensor, return_full_forecast=True)
        
        if 'multistep' in result:
            preds = result['multistep']['prices'].squeeze().cpu().numpy()
            uncer = result['multistep']['uncertainties'].squeeze().cpu().numpy()
            
            if preds.ndim == 0:
                predictions = [float(preds)]
                uncertainties = [float(uncer)]
            else:
                predictions = list(preds[:steps])
                uncertainties = list(uncer[:steps])
        else:
            # Fallback to single-step
            pred = result['price'].squeeze().cpu().numpy()
            if pred.ndim == 0:
                predictions = [float(pred)]
                uncertainties = [0.1]
            else:
                predictions = [float(pred[0])]
                uncertainties = [0.1]
    
    return np.array(predictions), np.array(uncertainties)


def visualize_optimized(symbol='SOL', lookback=60, predict_steps=5):
    """Enhanced optimization visualization"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    try:
        # Fetch and prepare data
        logger.info(f"[1/8] Fetching {symbol} data...")
        fetcher = TFTDataFetcher()
        df = fetcher.fetch_ohlcv_binance(f"{symbol}/USDT", timeframe='1h', limit=5000)
        
        if df is None or df.empty:
            logger.error(f"Failed to fetch data")
            return
        
        logger.info(f"[2/8] Adding indicators...")
        df = fetcher.add_tft_indicators(df)
        
        logger.info(f"[3/8] Preparing features...")
        X, y_original, scaler = fetcher.prepare_ml_features(df, lookback=lookback)
        
        if X is None:
            return
        
        # Load model
        logger.info(f"[4/8] Loading model...")
        model_paths = [
            f'models/saved_models/{symbol}_tft_multistep_best.pth',
            f'models/saved_models/{symbol}_tft_enhanced_model.pth',
            f'models/saved_models/{symbol}_tft_model.pth'
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            logger.error(f"Model not found")
            return
        
        input_size = X.shape[2]
        model = TemporalFusionTransformerV3EnhancedOptimized(
            input_size=input_size,
            hidden_size=256,
            num_heads=8,
            num_layers=3,
            dropout=0.2,
            output_size=1,
            forecast_steps=5,
            use_direction_head=True,
            use_multistep_head=True
        ).to(device)
        
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        logger.info(f"Model loaded: {model_path}")
        
        # Generate predictions
        logger.info(f"[5/8] Generating predictions...")
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            preds_scaled = model(X_tensor).squeeze().cpu().numpy()
        
        # Inverse transform
        num_features = X.shape[2]
        preds_full = np.zeros((len(preds_scaled), num_features))
        preds_full[:, 0] = preds_scaled
        preds_inverse = scaler.inverse_transform(preds_full)[:, 0]
        
        y_true = y_original
        y_pred_raw = preds_inverse
        
        # Ensemble smoothing
        logger.info(f"[6/8] Ensemble smoothing...")
        y_pred_ensemble, kalman, exponential, ma = OptimizedEnsembleSmoother.ensemble_smooth(y_pred_raw)
        
        # Volatility
        volatility = np.std(np.diff(X[:, :, 0], axis=1), axis=1)
        
        # Metrics
        logger.info(f"[7/8] Computing metrics...")
        metrics_raw = PerformanceMetrics.calculate(y_true, y_pred_raw, volatility)
        metrics_smooth = PerformanceMetrics.calculate(y_true, y_pred_ensemble, volatility)
        
        # Multi-step forecast
        logger.info(f"[8/8] Multi-step forecasting...")
        if len(X) > 0:
            future_preds_scaled, future_uncer = predict_multistep(
                model, X[[-1]], steps=predict_steps, device=device
            )
            future_full = np.zeros((len(future_preds_scaled), num_features))
            future_full[:, 0] = future_preds_scaled
            future_inverse = scaler.inverse_transform(future_full)[:, 0]
        else:
            future_inverse = np.zeros(predict_steps)
            future_uncer = np.ones(predict_steps) * 0.1
        
        # Print results
        logger.info(f"\n{'='*80}")
        logger.info(f"TFT V3 OPTIMIZED PERFORMANCE - {symbol} v1.2+")
        logger.info(f"{'='*80}\n")
        
        logger.info(f"RAW PREDICTIONS:")
        logger.info(f"  MAE:              {metrics_raw['mae']:.6f} USD")
        logger.info(f"  MAPE:             {metrics_raw['mape']:.4f}%")
        logger.info(f"  RMSE:             {metrics_raw['rmse']:.6f} USD")
        logger.info(f"  SMAPE:            {metrics_raw['smape']:.4f}%")
        logger.info(f"  R2:               {metrics_raw['r2']:.6f}")
        logger.info(f"  Dir. Accuracy:    {metrics_raw['dir_acc']:.2f}%\n")
        
        logger.info(f"ENSEMBLE SMOOTHED (V1.2 Optimized):")
        mae_status = "[GOOD]" if metrics_smooth['mae'] < metrics_raw['mae'] else "[X]"
        mape_status = "[GOOD]" if metrics_smooth['mape'] < metrics_raw['mape'] else "[X]"
        rmse_status = "[GOOD]" if metrics_smooth['rmse'] < metrics_raw['rmse'] else "[X]"
        dir_acc_status = "[GOOD]" if metrics_smooth['dir_acc'] > metrics_raw['dir_acc'] else "[X]"
        
        logger.info(f"  MAE:              {metrics_smooth['mae']:.6f} USD {mae_status}")
        logger.info(f"  MAPE:             {metrics_smooth['mape']:.4f}% {mape_status}")
        logger.info(f"  RMSE:             {metrics_smooth['rmse']:.6f} USD {rmse_status}")
        logger.info(f"  SMAPE:            {metrics_smooth['smape']:.4f}%")
        logger.info(f"  R2:               {metrics_smooth['r2']:.6f}")
        logger.info(f"  Dir. Accuracy:    {metrics_smooth['dir_acc']:.2f}% {dir_acc_status}\n")
        
        if metrics_smooth['mae_high_vol'] and metrics_smooth['mae_low_vol']:
            logger.info(f"VOLATILITY-ADJUSTED ACCURACY:")
            logger.info(f"  High Vol MAE:     {metrics_smooth['mae_high_vol']:.6f} USD")
            logger.info(f"  Low Vol MAE:      {metrics_smooth['mae_low_vol']:.6f} USD\n")
        
        logger.info(f"MULTI-STEP FORECAST ({predict_steps} Candles):")
        logger.info(f"  Current Price: {y_true[-1]:.6f} USD\n")
        for i, (price, uncer) in enumerate(zip(future_inverse, future_uncer), 1):
            if y_true[-1] != 0:
                change = ((price - y_true[-1]) / y_true[-1]) * 100
                logger.info(f"  +{i}h: {price:.6f} USD ({change:+.2f}%) [Conf: ±{uncer:.6f}]")
            else:
                logger.info(f"  +{i}h: {price:.6f} USD [Conf: ±{uncer:.6f}]")
        
        logger.info(f"\n{'='*80}")
        
        # Save metrics
        os.makedirs('models/training_logs', exist_ok=True)
        metrics_file = f'models/training_logs/{symbol}_metrics_v1.2_optimized.json'
        
        with open(metrics_file, 'w') as f:
            json.dump({
                'symbol': symbol,
                'version': 'v1.2_optimized',
                'timestamp': datetime.now().isoformat(),
                'raw_metrics': metrics_raw,
                'smoothed_metrics': metrics_smooth,
                'multistep_forecast': {
                    'prices': list(future_inverse),
                    'uncertainties': list(future_uncer)
                }
            }, f, indent=2)
        
        logger.info(f"Metrics saved: {metrics_file}")
        
        # Visualization
        logger.info(f"\nGenerating advanced visualization...")
        fig = plt.figure(figsize=(24, 16))
        fig.suptitle(
            f'{symbol} TFT V3 Optimized Analysis v1.2 - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            fontsize=18, fontweight='bold'
        )
        
        # Plot 1: Price with ensemble smoothing
        ax1 = plt.subplot(3, 4, 1)
        last_n = min(200, len(y_true))
        x_range = range(last_n)
        
        ax1.plot(x_range, y_true[-last_n:], 'b-', linewidth=2.5, alpha=0.9, label='Actual Price', marker='o', markersize=2)
        ax1.plot(x_range, y_pred_ensemble[-last_n:], 'g-', linewidth=2.5, alpha=0.9, label='Ensemble Pred')
        
        error_std = np.std(y_true[-last_n:] - y_pred_ensemble[-last_n:])
        ax1.fill_between(x_range,
                         y_pred_ensemble[-last_n:] - error_std,
                         y_pred_ensemble[-last_n:] + error_std,
                         alpha=0.15, color='green', label='Confidence')
        
        ax1.set_title(f'{symbol} Price Ensemble\nMAE: {metrics_smooth["mae"]:.6f} USD',
                     fontsize=11, fontweight='bold')
        ax1.set_xlabel('Hours')
        ax1.set_ylabel('Price (USD)')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Error distribution
        ax2 = plt.subplot(3, 4, 2)
        errors = y_pred_ensemble - y_true
        ax2.hist(errors, bins=60, color='purple', alpha=0.6, edgecolor='black')
        ax2.axvline(0, color='red', linestyle='--', linewidth=2)
        ax2.set_title('Error Distribution', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Error (USD)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Actual vs Predicted
        ax3 = plt.subplot(3, 4, 3)
        scatter = ax3.scatter(y_true, y_pred_ensemble, c=range(len(y_true)),
                             cmap='viridis', alpha=0.6, s=20)
        min_v = min(y_true.min(), y_pred_ensemble.min())
        max_v = max(y_true.max(), y_pred_ensemble.max())
        ax3.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2)
        ax3.set_title(f'Actual vs Predicted\nR2: {metrics_smooth["r2"]:.4f}',
                     fontsize=11, fontweight='bold')
        ax3.set_xlabel('Actual (USD)')
        ax3.set_ylabel('Predicted (USD)')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Error over time
        ax4 = plt.subplot(3, 4, 4)
        ax4.plot(errors, alpha=0.7, color='orange', linewidth=1)
        ax4.axhline(0, color='red', linestyle='--', alpha=0.6)
        ax4.fill_between(range(len(errors)), 0, errors, where=(errors >= 0),
                        alpha=0.2, color='red')
        ax4.fill_between(range(len(errors)), 0, errors, where=(errors < 0),
                        alpha=0.2, color='green')
        ax4.set_title('Prediction Error Over Time', fontsize=11, fontweight='bold')
        ax4.set_xlabel('Sample')
        ax4.set_ylabel('Error (USD)')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5-8: Metrics comparisons
        ax5 = plt.subplot(3, 4, 5)
        metrics_names = ['MAE', 'MAPE', 'RMSE', 'SMAPE']
        raw_vals = [metrics_raw['mae'], metrics_raw['mape'], metrics_raw['rmse'], metrics_raw['smape']]
        smooth_vals = [metrics_smooth['mae'], metrics_smooth['mape'], metrics_smooth['rmse'], metrics_smooth['smape']]
        x_pos = np.arange(len(metrics_names))
        width = 0.35
        ax5.bar(x_pos - width/2, raw_vals, width, label='Raw', color='red', alpha=0.7)
        ax5.bar(x_pos + width/2, smooth_vals, width, label='Ensemble', color='green', alpha=0.7)
        ax5.set_ylabel('Error')
        ax5.set_title('Metrics Comparison', fontsize=11, fontweight='bold')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(metrics_names, fontsize=9)
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Plot 6: Direction accuracy by volatility
        ax6 = plt.subplot(3, 4, 6)
        vol_bins = pd.qcut(volatility, q=4, duplicates='drop')
        dir_acc_by_vol = []
        for vol_range in vol_bins.unique():
            mask = vol_bins == vol_range
            if mask.sum() > 0:
                true_dir = np.sign(np.diff(y_true[mask]))
                pred_dir = np.sign(np.diff(y_pred_ensemble[mask]))
                acc = np.mean(true_dir == pred_dir) * 100 if len(true_dir) > 0 else 50
                dir_acc_by_vol.append(acc)
        ax6.bar(range(len(dir_acc_by_vol)), dir_acc_by_vol, color='steelblue', alpha=0.7)
        ax6.axhline(50, color='red', linestyle='--', alpha=0.7)
        ax6.set_xlabel('Volatility Quartile')
        ax6.set_ylabel('Direction Accuracy (%)')
        ax6.set_title('Accuracy vs Volatility', fontsize=11, fontweight='bold')
        ax6.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Plot 7: Multi-step forecast with confidence
        ax7 = plt.subplot(3, 4, 7)
        hist_len = 20
        hist_data = y_true[-hist_len:]
        hist_x = np.arange(hist_len)
        ax7.plot(hist_x, hist_data, 'o-', color='blue', label='Historical', linewidth=2)
        
        forecast_data = np.concatenate([[y_true[-1]], future_inverse])
        forecast_x = np.arange(hist_len - 1, hist_len - 1 + len(forecast_data))
        ax7.plot(forecast_x, forecast_data, 's--', color='green', label='Forecast', linewidth=2, markersize=8)
        
        # Confidence bands
        ax7.fill_between(forecast_x,
                        forecast_data - future_uncer,
                        forecast_data + future_uncer,
                        alpha=0.2, color='green', label='Uncertainty')
        
        ax7.set_title(f'Multi-Step Forecast ({predict_steps}h)', fontsize=11, fontweight='bold')
        ax7.set_xlabel('Time')
        ax7.set_ylabel('Price (USD)')
        ax7.legend(fontsize=9)
        ax7.grid(True, alpha=0.3)
        
        # Plot 8: Summary text
        ax8 = plt.subplot(3, 4, 8)
        ax8.axis('off')
        
        price_range = y_true.max() - y_true.min()
        mae_pct = (metrics_smooth['mae'] / y_true.mean()) * 100
        mape_pct = metrics_smooth['mape']
        
        summary_text = f"""TFT V3 OPTIMIZED v1.2+ SUMMARY

Performance:
  MAE: {metrics_smooth['mae']:.6f} USD
  MAE %: {mae_pct:.4f}%
  MAPE: {mape_pct:.4f}%
  RMSE: {metrics_smooth['rmse']:.6f} USD
  R²: {metrics_smooth['r2']:.6f}
  Dir Acc: {metrics_smooth['dir_acc']:.2f}%

Data:
  Samples: {len(y_true):,}
  Price: {y_true.min():.2f}-{y_true.max():.2f}
  Mean: {y_true.mean():.2f}
  Vol: {volatility.mean():.6f}

Forecast:
  Steps: {predict_steps}
  Confidence: ±{future_uncer[0]:.6f}

Status:
  {'✓ OPTIMIZED' if metrics_smooth['mae'] < metrics_raw['mae'] else '⚠ REVIEW'}
"""
        
        ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        output_dir = 'analysis_plots'
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(output_dir, f'{symbol}_tft_v3_optimized_{timestamp}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Plot saved: {save_path}")
        
        plt.show()
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}", exc_info=True)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='TFT V3 Optimized Visualization')
    parser.add_argument('--symbol', type=str, default='SOL', help='Crypto symbol')
    parser.add_argument('--lookback', type=int, default=60, help='Lookback period')
    parser.add_argument('--steps', type=int, default=5, help='Forecast steps')
    args = parser.parse_args()
    
    logger.info(f"\nTFT V3 Optimized Visualization v1.2+: {args.symbol}\n")
    visualize_optimized(args.symbol, args.lookback, args.steps)
