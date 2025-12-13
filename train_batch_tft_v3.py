#!/usr/bin/env python3
"""
🚀 Batch TFT Training V3 - Train 20+ Cryptocurrencies

✨ Features:
1. Batch training for multiple symbols
2. Parallel processing support
3. Progress tracking and summaries
4. Automatic model saving
5. Error handling and logging

📊 Usage:
  python train_batch_tft_v3.py --symbols BTC ETH SOL DOGE ...
  python train_batch_tft_v3.py --count 20  # Train top 20 by market cap
  python train_batch_tft_v3.py --list symbols.txt  # From file
"""

import argparse
import logging
from pathlib import Path
import sys
import torch
import torch.nn as nn
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# UTF-8 encoding support for Windows
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import setup_logging, create_directories
from src.data_fetcher_tft_v3 import TFTDataFetcher
from src.model_tft_v3 import TemporalFusionTransformer
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

load_dotenv()
setup_logging(log_level='INFO', log_file='logs/training_batch_tft_v3.log')
logger = logging.getLogger(__name__)


class WeightedMSELoss(nn.Module):
    """MSE Loss with temporal weighting"""
    def __init__(self, weight_power=2.0):
        super().__init__()
        self.weight_power = weight_power
    
    def forward(self, pred, target):
        if pred.dim() == 0:
            pred = pred.unsqueeze(0)
        if target.dim() == 0:
            target = target.unsqueeze(0)
        
        n_samples = pred.shape[0]
        if n_samples > 1:
            weights = torch.linspace(0.5, 1.5, n_samples, device=pred.device) ** self.weight_power
        else:
            weights = torch.ones(n_samples, device=pred.device)
        
        mse = (pred - target) ** 2
        weighted_mse = mse * weights.unsqueeze(-1)
        
        return weighted_mse.mean()


class DirectionalLoss(nn.Module):
    """Loss that encourages correct directional predictions"""
    def __init__(self, weight=0.3):
        super().__init__()
        self.weight = weight
    
    def forward(self, pred, target):
        if pred.dim() == 0 or target.dim() == 0:
            return torch.tensor(0.0, device=pred.device)
        
        if len(pred) > 1 and len(target) > 1:
            true_dir = torch.sign(target[1:] - target[:-1])
            pred_dir = torch.sign(pred[1:] - pred[:-1])
            
            direction_error = 1.0 - (true_dir * pred_dir + 1.0) / 2.0
            return direction_error.mean() * self.weight
        
        return torch.tensor(0.0, device=pred.device)


class TFTBatchTrainer:
    """Batch trainer for multiple cryptocurrencies"""
    
    def __init__(self, device='cuda', max_workers=2):
        self.device = device
        self.max_workers = max_workers
        self.results = {}
        self.fetcher = TFTDataFetcher()
    
    def train_single_symbol(self, symbol, epochs=100, batch_size=32, learning_rate=0.0001):
        """Train model for a single symbol"""
        try:
            logger.info(f"\n{'='*80}")
            logger.info(f"Training {symbol}")
            logger.info(f"{'='*80}")
            
            # Fetch data
            logger.info(f"[1/5] Fetching {symbol}/USDT data...")
            df = self.fetcher.fetch_ohlcv_binance(f"{symbol}/USDT", timeframe='1h', limit=3000)
            
            if df is None or len(df) < 100:
                logger.warning(f"Insufficient data for {symbol} (got {len(df) if df is not None else 0} candles)")
                return {'symbol': symbol, 'status': 'FAILED', 'reason': 'Insufficient data'}
            
            logger.info(f"Got {len(df)} candles")
            
            # Add indicators
            logger.info(f"[2/5] Adding indicators...")
            df = self.fetcher.add_tft_indicators(df)
            
            # Prepare features
            logger.info(f"[3/5] Preparing features...")
            X, y, scaler = self.fetcher.prepare_ml_features(df, lookback=60)
            
            if X is None or len(X) < 50:
                logger.warning(f"Failed to prepare features for {symbol}")
                return {'symbol': symbol, 'status': 'FAILED', 'reason': 'Feature preparation failed'}
            
            logger.info(f"Prepared {len(X)} samples with {X.shape[2]} features")
            
            # Initialize model
            logger.info(f"[4/5] Initializing model...")
            model = TemporalFusionTransformer(
                input_size=X.shape[2],
                hidden_size=128,  # Smaller for batch training
                num_heads=4,
                num_layers=2,
                dropout=0.2
            ).to(self.device)
            
            # Split data
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Create dataloaders
            train_dataset = TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32)
            )
            val_dataset = TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            
            # Setup training
            logger.info(f"[5/5] Training model...")
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
            
            loss_fns = nn.ModuleList([
                nn.MSELoss(),
                WeightedMSELoss(weight_power=1.5),
                DirectionalLoss(weight=0.2)
            ])
            
            best_val_loss = float('inf')
            patience_counter = 0
            best_model_state = None
            
            # Training loop
            for epoch in range(epochs):
                model.train()
                train_loss = 0.0
                
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device).unsqueeze(-1)
                    
                    optimizer.zero_grad()
                    predictions = model(X_batch)
                    
                    if predictions.shape != y_batch.shape:
                        predictions = predictions.squeeze(-1).unsqueeze(-1)
                    
                    loss = sum(fn(predictions, y_batch) for fn in loss_fns)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                train_loss /= len(train_loader)
                
                # Validation
                model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device).unsqueeze(-1)
                        
                        predictions = model(X_batch)
                        if predictions.shape != y_batch.shape:
                            predictions = predictions.squeeze(-1).unsqueeze(-1)
                        
                        loss = sum(fn(predictions, y_batch) for fn in loss_fns)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                scheduler.step()
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if (epoch + 1) % 20 == 0:
                    logger.info(f"  Epoch {epoch+1:3d}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
                
                if patience_counter >= 15:
                    logger.info(f"  Early stopping at epoch {epoch+1}")
                    break
            
            # Restore best model
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            
            # Save model
            model_dir = Path('models/saved_models')
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f'{symbol}_tft_model.pth'
            torch.save(model.state_dict(), model_path)
            
            logger.info(f"Model saved to {model_path}")
            logger.info(f"Best Val Loss: {best_val_loss:.6f}")
            
            return {
                'symbol': symbol,
                'status': 'SUCCESS',
                'val_loss': float(best_val_loss),
                'epochs_trained': epoch + 1,
                'model_path': str(model_path)
            }
            
        except Exception as e:
            logger.error(f"Error training {symbol}: {e}", exc_info=True)
            return {
                'symbol': symbol,
                'status': 'FAILED',
                'error': str(e)
            }
    
    def train_batch(self, symbols, epochs=100, batch_size=32, learning_rate=0.0001, sequential=False):
        """Train multiple symbols"""
        logger.info(f"\n{'='*80}")
        logger.info(f"BATCH TRAINING - {len(symbols)} SYMBOLS")
        logger.info(f"{'='*80}")
        logger.info(f"Symbols: {', '.join(symbols)}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Sequential: {sequential}")
        logger.info(f"{'='*80}\n")
        
        results = []
        start_time = time.time()
        
        if sequential:
            # Sequential training
            for idx, symbol in enumerate(symbols, 1):
                logger.info(f"\n[{idx}/{len(symbols)}] Training {symbol}...")
                result = self.train_single_symbol(symbol, epochs, batch_size, learning_rate)
                results.append(result)
        else:
            # Parallel training (limited workers to avoid memory issues)
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self.train_single_symbol, symbol, epochs, batch_size, learning_rate): symbol 
                          for symbol in symbols}
                
                for idx, future in enumerate(as_completed(futures), 1):
                    try:
                        result = future.result()
                        results.append(result)
                        logger.info(f"\n[{idx}/{len(symbols)}] Completed: {result['symbol']} - Status: {result['status']}")
                    except Exception as e:
                        logger.error(f"Error: {e}")
        
        elapsed_time = time.time() - start_time
        
        # Summary
        logger.info(f"\n{'='*80}")
        logger.info(f"BATCH TRAINING SUMMARY")
        logger.info(f"{'='*80}")
        
        successful = [r for r in results if r['status'] == 'SUCCESS']
        failed = [r for r in results if r['status'] != 'SUCCESS']
        
        logger.info(f"\nTotal Symbols: {len(symbols)}")
        logger.info(f"Successful: {len(successful)}")
        logger.info(f"Failed: {len(failed)}")
        logger.info(f"Total Time: {elapsed_time/3600:.1f} hours")
        logger.info(f"Average Time per Symbol: {elapsed_time/len(symbols)/60:.1f} minutes")
        
        if successful:
            logger.info(f"\nSuccessful Models:")
            for r in successful:
                logger.info(f"  {r['symbol']:6s} - Val Loss: {r.get('val_loss', 'N/A'):.6f} - Epochs: {r.get('epochs_trained', 'N/A')}")
        
        if failed:
            logger.info(f"\nFailed Models:")
            for r in failed:
                logger.info(f"  {r['symbol']:6s} - Reason: {r.get('reason', r.get('error', 'Unknown'))}")
        
        logger.info(f"{'='*80}")
        
        # Save results
        results_file = Path('results/batch_training_results.json')
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {results_file}")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description='Batch TFT Training - Train multiple cryptocurrencies'
    )
    
    parser.add_argument('--symbols', nargs='+', default=['BTC', 'ETH', 'SOL', 'DOGE', 'XRP'],
                       help='List of symbols to train')
    parser.add_argument('--count', type=int, help='Train top N cryptocurrencies')
    parser.add_argument('--list', type=str, help='File with symbols (one per line)')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs per symbol')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--sequential', action='store_true', help='Train sequentially instead of parallel')
    parser.add_argument('--workers', type=int, default=2, help='Max parallel workers')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Get symbols
    symbols = args.symbols
    
    if args.list:
        with open(args.list, 'r') as f:
            symbols = [line.strip().upper() for line in f if line.strip()]
    
    if args.count:
        # Top cryptocurrencies by market cap
        top_symbols = ['BTC', 'ETH', 'SOL', 'DOGE', 'XRP', 'ADA', 'AVAX', 'LINK', 'MATIC', 'ARB',
                      'OP', 'LDO', 'SUI', 'NEAR', 'INJ', 'SEI', 'TON', 'FET', 'ICP', 'BLUR']
        symbols = top_symbols[:args.count]
    
    try:
        create_directories()
        
        trainer = TFTBatchTrainer(device=device, max_workers=args.workers)
        results = trainer.train_batch(
            symbols=symbols,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            sequential=args.sequential
        )
        
        logger.info("\nBatch training completed!")
        return True
        
    except Exception as e:
        logger.error(f"Batch training failed: {e}", exc_info=True)
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
