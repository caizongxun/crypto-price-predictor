#!/usr/bin/env python
"""Compare different model architectures intelligently."""

import os
import sys
import re
import torch
import numpy as np
import ccxt
import pandas as pd
from pathlib import Path
import logging

from src.signal_generator import SignalGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UniversalModelLoader:
    """Load models regardless of their architecture."""
    
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🖥️ Using device: {self.device}")

    def detect_architecture(self, model_path: str) -> str:
        """Detect model architecture from saved state_dict."""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            keys = set(state_dict.keys())
            
            # Check for ensemble/fusion indicators
            if any('lstm_model.' in k or 'gru_model.' in k or 'transformer_model.' in k for k in keys):
                return 'ensemble'
            elif 'lstm.' in str(keys):
                return 'lstm'
            else:
                return 'unknown'
        except:
            return 'unknown'

    def load_and_infer(self, model_path: str):
        """Load model and run inference without knowing exact architecture."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict):
                # Could be {'model_state_dict': ...} or direct state dict
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            arch = self.detect_architecture(model_path)
            logger.debug(f"Detected architecture: {arch}")
            
            # For ensemble models, we can't easily extract the model structure
            # But we can try to create a wrapper that extracts just the main model
            if arch == 'ensemble':
                # Ensemble models are too complex, skip them
                logger.warning(f"⚠️  Ensemble model detected, skipping: {Path(model_path).name}")
                return None, arch
            
            # For simple LSTM models
            from src.model_trainer import LSTMModel
            hidden_size = self._detect_hidden_size(state_dict)
            num_layers = self._detect_num_layers(state_dict)
            
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
            return model, arch
            
        except Exception as e:
            logger.debug(f"Load error: {e}")
            return None, 'unknown'

    def _detect_hidden_size(self, state_dict) -> int:
        """Detect hidden size from state dict."""
        for key in ['lstm.weight_ih_l0', 'lstm_model.lstm.weight_ih_l0']:
            if key in state_dict:
                return state_dict[key].shape[0] // 4
        return 256  # Default

    def _detect_num_layers(self, state_dict) -> int:
        """Detect number of layers from state dict."""
        num_layers = 0
        for i in range(10):
            if f'lstm.weight_ih_l{i}' in state_dict:
                num_layers = i + 1
        return max(num_layers, 2)  # Default to 2


class GroupComparator:
    """Compare two groups of models."""
    
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loader = UniversalModelLoader(device)
        self.exchange = ccxt.binance({'enableRateLimit': True})

    def fetch_data(self, symbol: str) -> pd.DataFrame:
        """Fetch OHLCV data."""
        try:
            ohlcv = self.exchange.fetch_ohlcv(f"{symbol}/USDT", timeframe='1h', limit=500)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            return df if len(df) > 100 else None
        except:
            return None

    def evaluate_model(self, model_path: str):
        """Evaluate a single model."""
        filename = Path(model_path).name
        
        # Extract symbol
        match = re.match(r"([A-Z]+)_lstm_model\.pth", filename)
        if not match:
            return None, 'unknown'
        
        symbol = match.group(1)
        model, arch = self.loader.load_and_infer(model_path)
        
        if not model:
            return None, arch
        
        # Fetch & evaluate
        df = self.fetch_data(symbol)
        if df is None:
            return None, arch
        
        gen = SignalGenerator()
        raw_features = gen.prepare_features(df['close'].values)
        
        X_test, y_test = [], []
        for i in range(len(raw_features) - 65):
            X_test.append(raw_features[i:i+60])
            y_test.append(raw_features[i+60:i+65, 3])
        
        if not X_test:
            return None, arch
        
        X_tensor = torch.from_numpy(np.array(X_test)).float().to(self.device)
        y_true = np.array(y_test)
        
        with torch.no_grad():
            pred = model(X_tensor).cpu().numpy()
            if pred.ndim > 2:
                pred = pred[:, -1, :]
        
        mae = np.mean(np.abs(y_true - pred))
        return {
            'symbol': symbol,
            'mae': mae,
            'size_mb': os.path.getsize(model_path) / (1024*1024),
            'arch': arch
        }, arch

    def run(self):
        """Run full comparison."""
        groups = {
            'NEW': list(Path('models/saved_models').glob('*_lstm_model.pth')),
            'OLD': list(Path('models/saved_models_old').glob('*_lstm_model.pth'))
        }
        
        results = {'NEW': [], 'OLD': []}
        
        print("\n" + "="*80)
        print(f"{'GROUP':<6} {'SYMBOL':<8} {'SIZE(MB)':<10} {'MAE':<12} {'ARCH':<10} {'STATUS'}")
        print("-" * 80)
        
        for group_name, files in groups.items():
            for f in sorted(files):
                res, arch = self.evaluate_model(str(f))
                if res:
                    results[group_name].append(res)
                    print(f"{group_name:<6} {res['symbol']:<8} {res['size_mb']:<10.2f} {res['mae']:<12.6f} {arch:<10} ✅")
                else:
                    print(f"{group_name:<6} {f.stem:<8} {'-':<10} {'-':<12} {arch:<10} ❌")
        
        # Summary
        print("\n" + "="*80)
        print("📊 SUMMARY")
        print("="*80)
        
        if results['NEW']:
            avg_new = np.mean([r['mae'] for r in results['NEW']])
            print(f"NEW: {len(results['NEW'])} models, avg MAE = {avg_new:.6f}")
        else:
            print(f"NEW: No successfully loaded models")
        
        if results['OLD']:
            avg_old = np.mean([r['mae'] for r in results['OLD']])
            print(f"OLD: {len(results['OLD'])} models, avg MAE = {avg_old:.6f}")
        else:
            print(f"OLD: No successfully loaded models")
        
        print("="*80 + "\n")

if __name__ == "__main__":
    GroupComparator().run()
