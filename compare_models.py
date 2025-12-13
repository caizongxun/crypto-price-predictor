#!/usr/bin/env python
"""Compare accuracy of new vs old trained models.

Usage:
    python compare_models.py
"""

import os
import sys
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


class ModelComparator:
    """Compare new and old model versions."""
    
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        logger.info(f"🖥️  Using device: {self.device}")
        
        # Initialize exchange for public data (no keys needed)
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        self.results = []
        
    def detect_hidden_size(self, model_path: str) -> int:
        """Auto-detect hidden_size from model weights."""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # LSTM weight shape: [hidden_size * 4, input_or_hidden_size]
            # First layer: lstm.weight_ih_l0 has shape [hidden_size*4, 17]
            if 'lstm.weight_ih_l0' in state_dict:
                weight_shape = state_dict['lstm.weight_ih_l0'].shape[0]
                hidden_size = weight_shape // 4
                logger.debug(f"Detected hidden_size: {hidden_size}")
                return hidden_size
        except Exception as e:
            logger.debug(f"Error detecting hidden size: {e}")
        
        return None

    def load_model_with_retry(self, model_path: str) -> LSTMModel:
        """Try loading model with different configurations."""
        if not os.path.exists(model_path):
            return None
        
        # First, try to auto-detect hidden size
        detected_hidden_size = self.detect_hidden_size(model_path)
        if detected_hidden_size:
            model = self._create_and_load(model_path, hidden_size=detected_hidden_size)
            if model is not None:
                return model
        
        # If detection failed, try common sizes
        for hidden_size in [128, 256, 512, 1024]:
            try:
                model = self._create_and_load(model_path, hidden_size=hidden_size)
                return model
            except RuntimeError:
                continue
        
        logger.error(f"❌ Could not load model: {Path(model_path).name}")
        return None

    def _create_and_load(self, model_path, hidden_size):
        """Helper to create and load specific model config."""
        model = LSTMModel(
            input_size=17,
            hidden_size=hidden_size,
            num_layers=3,
            output_size=5,
            dropout=0.3
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model
    
    def fetch_data(self, symbol: str, limit: int = 500) -> pd.DataFrame:
        """Fetch OHLCV data without API keys."""
        try:
            symbol_pair = f"{symbol}/USDT"
            ohlcv = self.exchange.fetch_ohlcv(symbol_pair, timeframe='1h', limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Calculate accuracy metrics."""
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {'mae': mae, 'rmse': rmse, 'r2': r2}
    
    def test_symbol(self, symbol: str, new_model_path: str, old_model_path: str = None):
        """Test both model versions for a symbol."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {symbol}...")
        
        # Load models
        new_model = self.load_model_with_retry(new_model_path)
        if new_model is None:
            logger.warning(f"⚠️  Skipping {symbol} (New model load failed)")
            return
            
        old_model = None
        if old_model_path:
            old_model = self.load_model_with_retry(old_model_path)
        
        # Fetch Data
        df = self.fetch_data(symbol)
        if df is None: return

        # Prepare Data
        try:
            gen = SignalGenerator()
            raw_features = gen.prepare_features(df['close'].values)
            
            # Create sequences
            LOOKBACK = 60
            HORIZON = 5
            X_test, y_test = [], []
            
            for i in range(len(raw_features) - LOOKBACK - HORIZON):
                X_test.append(raw_features[i : i + LOOKBACK])
                y_test.append(raw_features[i + LOOKBACK : i + LOOKBACK + HORIZON, 3]) # 3 is close price
            
            if not X_test: return
            
            X_test = torch.from_numpy(np.array(X_test)).float().to(self.device)
            y_test = torch.from_numpy(np.array(y_test)).float().to(self.device)
            
            # Predict
            with torch.no_grad():
                new_pred = new_model(X_test).cpu().numpy()
                if new_pred.ndim > 2: new_pred = new_pred[:, -1, :]
                
                old_pred = None
                if old_model:
                    old_pred = old_model(X_test).cpu().numpy()
                    if old_pred.ndim > 2: old_pred = old_pred[:, -1, :]

            # Metrics
            y_test_np = y_test.cpu().numpy()
            new_metrics = self.calculate_metrics(y_test_np, new_pred)
            
            result = {
                'symbol': symbol,
                'new_metrics': new_metrics,
                'old_metrics': None,
                'improvement': None
            }
            
            logger.info(f"📊 New Model MAE: {new_metrics['mae']:.6f}")
            
            if old_pred is not None:
                old_metrics = self.calculate_metrics(y_test_np, old_pred)
                result['old_metrics'] = old_metrics
                
                imp = ((old_metrics['mae'] - new_metrics['mae']) / old_metrics['mae']) * 100
                result['improvement'] = imp
                logger.info(f"📊 Old Model MAE: {old_metrics['mae']:.6f}")
                logger.info(f"📈 Improvement: {imp:+.2f}%")
                
            self.results.append(result)
            
        except Exception as e:
            logger.error(f"Error testing {symbol}: {e}")

    def generate_report(self):
        """Generate simple report."""
        if not self.results: return
        
        print("\n" + "="*60)
        print(f"{'SYMBOL':<10} {'NEW MAE':<12} {'OLD MAE':<12} {'IMPROVEMENT':<12}")
        print("-" * 60)
        
        for r in self.results:
            imp = f"{r['improvement']:+.2f}%" if r['improvement'] else "N/A"
            old = f"{r['old_metrics']['mae']:.6f}" if r['old_metrics'] else "N/A"
            print(f"{r['symbol']:<10} {r['new_metrics']['mae']:<12.6f} {old:<12} {imp:<12}")
        print("="*60 + "\n")

def main():
    comparator = ModelComparator()
    
    symbols = ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOGE', 'MATIC', 'AVAX', 'LINK']
    new_dir = Path('models/saved_models')
    old_dir = Path('models/saved_models_old')
    
    for symbol in symbols:
        new_path = new_dir / f"{symbol}_lstm_model.pth"
        old_path = old_dir / f"{symbol}_lstm_model.pth"
        
        if new_path.exists():
            comparator.test_symbol(symbol, str(new_path), str(old_path) if old_path.exists() else None)
            
    comparator.generate_report()

if __name__ == "__main__":
    main()
