#!/usr/bin/env python
"""Visualize predictions vs actual prices to analyze accuracy improvements."""

import os
import sys
import torch
import numpy as np
import pandas as pd
import ccxt
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.signal_generator import SignalGenerator
from src.model_trainer import LSTMModel

class ModelVisualizer:
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})
        self.output_dir = Path('analysis_plots')
        self.output_dir.mkdir(exist_ok=True)
        print(f"Device: {self.device}")
        
    def detect_model_config(self, state_dict):
        """Auto-detect hidden_size and num_layers from state dict."""
        # Detect hidden_size from weight shape
        if 'lstm.weight_ih_l0' in state_dict:
            hidden_size = state_dict['lstm.weight_ih_l0'].shape[0] // 4
        else:
            hidden_size = 256
            
        # Detect num_layers by checking which layer indices exist
        num_layers = 0
        for i in range(10):  # Check up to 10 layers
            if f'lstm.weight_ih_l{i}' in state_dict:
                num_layers = i + 1
            else:
                break
        num_layers = max(num_layers, 2)  # Default to 2 if detection fails
        
        print(f"  Detected: hidden_size={hidden_size}, num_layers={num_layers}")
        return hidden_size, num_layers
        
    def load_model(self, symbol):
        """Load the specific LSTM model for a symbol."""
        model_path = Path(f'models/saved_models/{symbol}_lstm_model.pth')
        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            return None
            
        print(f"Loading {symbol}...", end=' ')
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
            
            # Auto-detect config
            hidden_size, num_layers = self.detect_model_config(state_dict)
            
            model = LSTMModel(
                input_size=17,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=5,
                dropout=0.3
            )
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            print("✅")
            return model
        except Exception as e:
            print(f"❌ Error: {e}")
            return None

    def fetch_and_prepare_data(self, symbol):
        """Fetch data and prepare sequences."""
        try:
            ohlcv = self.exchange.fetch_ohlcv(f"{symbol}/USDT", timeframe='1h', limit=200)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            gen = SignalGenerator()
            raw_features = gen.prepare_features(df['close'].values)
            
            X, y, dates = [], [], []
            for i in range(len(raw_features) - 65):
                X.append(raw_features[i:i+60])
                y.append(raw_features[i+60:i+65, 3])  # Target is 'close' price
                dates.append(df['timestamp'].iloc[i+60])
                
            return np.array(X), np.array(y), dates, df
        except Exception as e:
            print(f"  Data fetch error: {e}")
            return None, None, None, None

    def plot_predictions(self, symbol):
        """Generate prediction vs actual plot."""
        model = self.load_model(symbol)
        if not model:
            return
        
        print(f"  Preparing data...", end=' ')
        X, y_true, dates, df = self.fetch_and_prepare_data(symbol)
        if X is None:
            print("❌")
            return
        print(f"✅ ({len(X)} sequences)")
        
        print(f"  Making predictions...", end=' ')
        X_tensor = torch.from_numpy(X).float().to(self.device)
        
        with torch.no_grad():
            preds = model(X_tensor).cpu().numpy()
            if preds.ndim > 2:
                preds = preds[:, -1, :]  # Take last step if sequence
            
        # Extract 1st step prediction (next 1 hour)
        y_true_next_1h = y_true[:, 0]
        preds_next_1h = preds[:, 0]
        
        mae = np.mean(np.abs(y_true_next_1h - preds_next_1h))
        print(f"✅ (MAE: {mae:.4f})")
        
        # Plot
        print(f"  Plotting...", end=' ')
        plt.figure(figsize=(14, 6))
        
        # Main plot
        plt.subplot(1, 2, 1)
        plt.plot(dates, y_true_next_1h, label='Actual (Normalized)', color='blue', alpha=0.8, linewidth=2)
        plt.plot(dates, preds_next_1h, label='Predicted (Normalized)', color='orange', alpha=0.8, linestyle='--', linewidth=2)
        plt.title(f'{symbol} Price Prediction (1h Ahead)\nMAE: {mae:.4f}', fontsize=12, fontweight='bold')
        plt.xlabel('Time')
        plt.ylabel('Normalized Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Error distribution
        plt.subplot(1, 2, 2)
        errors = np.abs(y_true_next_1h - preds_next_1h)
        plt.hist(errors, bins=30, color='coral', alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, label=f'Mean Error: {np.mean(errors):.4f}')
        plt.title(f'{symbol} Error Distribution')
        plt.xlabel('Absolute Error')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.output_dir / f'{symbol}_prediction_analysis.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"✅")
        print(f"  Saved: {save_path}")
        plt.close()

    def run_analysis(self):
        """Analyze top performers and worst performers."""
        # Analyze: Best (MATIC), Good (ETH), Worst (SOL), Medium (BTC)
        symbols = ['MATIC', 'ETH', 'BTC', 'SOL']
        
        print("\n" + "="*60)
        print("🎨 GENERATING PREDICTION ANALYSIS PLOTS")
        print("="*60)
        
        for s in symbols:
            self.plot_predictions(s)
        
        print("\n" + "="*60)
        print(f"✅ Analysis complete! Check {self.output_dir}/ for plots")
        print("="*60 + "\n")

if __name__ == "__main__":
    ModelVisualizer().run_analysis()
