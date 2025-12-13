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
        
    def load_model(self, symbol):
        """Load the specific LSTM model for a symbol."""
        model_path = Path(f'models/saved_models/{symbol}_lstm_model.pth')
        if not model_path.exists():
            return None
            
        # Detect hidden size (from previous analysis we know it's likely 256 or 128)
        # We'll use a robust loading method
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
        
        # Auto-detect hidden size
        hidden_size = 256
        if 'lstm.weight_ih_l0' in state_dict:
            hidden_size = state_dict['lstm.weight_ih_l0'].shape[0] // 4
            
        model = LSTMModel(input_size=17, hidden_size=hidden_size, num_layers=3, output_size=5, dropout=0.3)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def fetch_and_prepare_data(self, symbol):
        """Fetch data and prepare sequences."""
        print(f"Fetching data for {symbol}...")
        try:
            ohlcv = self.exchange.fetch_ohlcv(f"{symbol}/USDT", timeframe='1h', limit=200)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            gen = SignalGenerator()
            raw_features = gen.prepare_features(df['close'].values)
            
            X, y, dates = [], [], []
            for i in range(len(raw_features) - 65):
                X.append(raw_features[i:i+60])
                y.append(raw_features[i+60:i+65, 3]) # Target is 'close' price
                dates.append(df['timestamp'].iloc[i+60])
                
            return np.array(X), np.array(y), dates, df
        except Exception as e:
            print(f"Error: {e}")
            return None, None, None, None

    def plot_predictions(self, symbol):
        """Generate prediction vs actual plot."""
        model = self.load_model(symbol)
        if not model: return
        
        X, y_true, dates, df = self.fetch_and_prepare_data(symbol)
        if X is None: return
        
        X_tensor = torch.from_numpy(X).float().to(self.device)
        
        with torch.no_grad():
            preds = model(X_tensor).cpu().numpy()
            if preds.ndim > 2: preds = preds[:, -1, :] # Take last step if sequence
            
        # We only plot the 1st step prediction for clarity (next 1 hour)
        y_true_next_1h = y_true[:, 0]
        preds_next_1h = preds[:, 0]
        
        # Inverse transform (approximate) to get real prices
        # Since we used MinMaxScaler in training (implied), we need to check how scaling was done.
        # For visualization, normalized comparison is often enough, but let's try to align them.
        # Here we just plot normalized values to see the TREND matching.
        
        plt.figure(figsize=(12, 6))
        plt.plot(dates, y_true_next_1h, label='Actual (Normalized)', color='blue', alpha=0.7)
        plt.plot(dates, preds_next_1h, label='Predicted (Normalized)', color='orange', alpha=0.7, linestyle='--')
        
        mae = np.mean(np.abs(y_true_next_1h - preds_next_1h))
        plt.title(f'{symbol} Prediction Analysis (MAE: {mae:.4f})')
        plt.xlabel('Time')
        plt.ylabel('Normalized Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = self.output_dir / f'{symbol}_prediction_analysis.png'
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
        plt.close()

    def run_analysis(self):
        symbols = ['BTC', 'ETH', 'SOL', 'MATIC'] # Analyze top and worst performers
        for s in symbols:
            self.plot_predictions(s)

if __name__ == "__main__":
    ModelVisualizer().run_analysis()
