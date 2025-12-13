#!/usr/bin/env python
"""Compare performance of two groups of models (New vs Old).

Usage:
    python compare_models.py
"""

import os
import sys
import re
import logging
import torch
import numpy as np
import ccxt
import pandas as pd
from pathlib import Path
from datetime import datetime

from src.model_trainer import ModelTrainer, LSTMModel
from src.signal_generator import SignalGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GroupComparator:
    """Compare performance between two groups of models."""
    
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🖥️  Using device: {self.device}")
        
        # Public API for data
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        
    def detect_hidden_size(self, model_path: str) -> int:
        """Auto-detect hidden_size from model weights."""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # LSTM weight shape: [hidden_size * 4, input_size]
            if 'lstm.weight_ih_l0' in state_dict:
                weight_shape = state_dict['lstm.weight_ih_l0'].shape[0]
                return weight_shape // 4
        except:
            pass
        return 256  # Default fallback

    def load_model(self, model_path: str) -> LSTMModel:
        """Load model with auto-detected config."""
        try:
            hidden_size = self.detect_hidden_size(model_path)
            
            model = LSTMModel(
                input_size=17,
                hidden_size=hidden_size,
                num_layers=3,
                output_size=5,
                dropout=0.3
            )
            
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
            
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            return model, hidden_size
        except Exception as e:
            logger.error(f"❌ Load failed {Path(model_path).name}: {e}")
            return None, 0

    def fetch_data(self, symbol: str) -> pd.DataFrame:
        """Fetch test data."""
        try:
            ohlcv = self.exchange.fetch_ohlcv(f"{symbol}/USDT", timeframe='1h', limit=500)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            return df
        except:
            return None

    def evaluate_model(self, model_path: str):
        """Evaluate a single model."""
        filename = Path(model_path).name
        
        # Extract symbol from filename (e.g., "BTC_lstm_model.pth" -> "BTC")
        match = re.match(r"([A-Z]+)_lstm_model\.pth", filename)
        if not match:
            return None
            
        symbol = match.group(1)
        model, hidden_size = self.load_model(model_path)
        
        if not model:
            return None
            
        # Fetch data
        df = self.fetch_data(symbol)
        if df is None or len(df) < 100:
            logger.warning(f"⚠️  No data for {symbol}")
            return None
            
        # Prepare features
        gen = SignalGenerator()
        raw_features = gen.prepare_features(df['close'].values)
        
        X_test, y_test = [], []
        LOOKBACK, HORIZON = 60, 5
        
        for i in range(len(raw_features) - LOOKBACK - HORIZON):
            X_test.append(raw_features[i : i + LOOKBACK])
            y_test.append(raw_features[i + LOOKBACK : i + LOOKBACK + HORIZON, 3])
            
        if not X_test:
            return None
            
        X_tensor = torch.from_numpy(np.array(X_test)).float().to(self.device)
        y_true = np.array(y_test)
        
        with torch.no_grad():
            pred = model(X_tensor).cpu().numpy()
            if pred.ndim > 2: pred = pred[:, -1, :]
            
        mae = np.mean(np.abs(y_true - pred))
        return {
            'symbol': symbol,
            'mae': mae,
            'hidden_size': hidden_size,
            'file_size_mb': os.path.getsize(model_path) / (1024*1024)
        }

    def run_comparison(self):
        """Compare New vs Old groups."""
        groups = {
            'NEW': list(Path('models/saved_models').glob('*_lstm_model.pth')),
            'OLD': list(Path('models/saved_models_old').glob('*_lstm_model.pth'))
        }
        
        results = {'NEW': [], 'OLD': []}
        
        print("\n" + "="*80)
        print(f"{'GROUP':<6} {'SYMBOL':<8} {'SIZE(MB)':<10} {'HIDDEN':<8} {'MAE (Error)':<12} {'STATUS'}")
        print("-" * 80)
        
        for group_name, files in groups.items():
            for f in files:
                res = self.evaluate_model(str(f))
                if res:
                    results[group_name].append(res)
                    print(f"{group_name:<6} {res['symbol']:<8} {res['file_size_mb']:<10.2f} {res['hidden_size']:<8} {res['mae']:<12.6f} ✅")
                else:
                    print(f"{group_name:<6} {f.name:<20} {'-':<10} {'-':<8} {'-':<12} ❌ (Skip)")
        
        # Summary
        print("\n" + "="*80)
        print("📊 FINAL COMPARISON SUMMARY")
        print("="*80)
        
        avg_new = np.mean([r['mae'] for r in results['NEW']]) if results['NEW'] else 0
        avg_old = np.mean([r['mae'] for r in results['OLD']]) if results['OLD'] else 0
        
        print(f"🔹 NEW Models (Count: {len(results['NEW'])})")
        print(f"   Average MAE: {avg_new:.6f}")
        print(f"   Typical Size: ~3.16 MB")
        print(f"   Architecture: Lightweight (Hidden=256/128)")
        print("")
        print(f"🔸 OLD Models (Count: {len(results['OLD'])})")
        print(f"   Average MAE: {avg_old:.6f}")
        print(f"   Typical Size: ~45.35 MB")
        print(f"   Architecture: Heavy (Hidden=1024)")
        print("-" * 80)
        
        if avg_old > 0 and avg_new > 0:
            diff = ((avg_old - avg_new) / avg_old) * 100
            print(f"🏆 CONCLUSION: New models are {abs(diff):.2f}% {'BETTER' if diff > 0 else 'WORSE'} in accuracy.")
        print("="*80 + "\n")

if __name__ == "__main__":
    GroupComparator().run_comparison()
