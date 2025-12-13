#!/usr/bin/env python
"""Compare accuracy of new vs old trained models.

Usage:
    python compare_models.py

This script:
    1. Loads test data for each cryptocurrency
    2. Tests both old and new model versions
    3. Calculates accuracy metrics (MAE, RMSE, R²)
    4. Generates comparison report

Required directories:
    - models/saved_models/         (new models: 3MB versions)
    - models/saved_models_old/     (old models: 45MB versions)
"""

import os
import sys
import logging
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

from src.model_trainer import ModelTrainer, LSTMModel
from src.data_fetcher import DataFetcher
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
        """Initialize comparator.
        
        Args:
            device: torch device (cpu or cuda), auto-detect if None
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        logger.info(f"🖥️  Using device: {self.device}")
        
        self.data_fetcher = DataFetcher()
        self.results = []
        
    def load_model(self, model_path: str) -> LSTMModel:
        """Load a trained model.
        
        Args:
            model_path: Path to model .pth file
            
        Returns:
            Loaded PyTorch model or None if failed
        """
        try:
            if not os.path.exists(model_path):
                logger.warning(f"⚠️  Model not found: {model_path}")
                return None
                
            trainer = ModelTrainer(model_type='lstm', config={
                'hidden_size': 256,
                'num_layers': 3,
                'dropout': 0.3
            })
            
            trainer.load_model(model_path, input_size=17)
            trainer.model.to(self.device)
            trainer.model.eval()
            
            return trainer.model
            
        except Exception as e:
            logger.error(f"Error loading model {model_path}: {e}")
            return None
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Calculate accuracy metrics.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            Dictionary with MAE, RMSE, R² metrics
        """
        # Mean Absolute Error
        mae = np.mean(np.abs(y_true - y_pred))
        
        # Root Mean Squared Error
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        
        # R² Score (coefficient of determination)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else np.inf
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
    
    def test_symbol(self, symbol: str, new_model_path: str, old_model_path: str = None) -> dict:
        """Test both model versions for a symbol.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
            new_model_path: Path to new model
            old_model_path: Path to old model (optional)
            
        Returns:
            Dictionary with comparison results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {symbol}...")
        logger.info(f"{'='*60}")
        
        # Load models
        new_model = self.load_model(new_model_path)
        if new_model is None:
            logger.error(f"❌ Cannot load new model for {symbol}")
            return None
        
        old_model = None
        if old_model_path:
            old_model = self.load_model(old_model_path)
        
        # Fetch test data
        try:
            trading_pair = f"{symbol}/USDT"
            df = self.data_fetcher.fetch_ohlcv_binance(trading_pair, '1h', limit=500)
            
            if df is None or len(df) < 100:
                logger.warning(f"⚠️  Insufficient data for {symbol}")
                return None
            
            logger.info(f"✅ Fetched {len(df)} data points for {symbol}")
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
        
        # Prepare features
        try:
            from src.signal_generator import SignalGenerator
            gen = SignalGenerator()
            raw_features = gen.prepare_features(df['close'].values)
            
            if raw_features is None or len(raw_features) < 65:
                logger.warning(f"⚠️  Insufficient features for {symbol}")
                return None
            
            # Create sequences
            LOOKBACK = 60
            HORIZON = 5
            
            X_test = []
            y_test = []
            close_col_idx = 3
            
            for i in range(len(raw_features) - LOOKBACK - HORIZON):
                X_test.append(raw_features[i : i + LOOKBACK])
                y_test.append(raw_features[i + LOOKBACK : i + LOOKBACK + HORIZON, close_col_idx])
            
            if len(X_test) == 0:
                logger.warning(f"⚠️  No test sequences for {symbol}")
                return None
            
            X_test = torch.from_numpy(np.array(X_test)).float().to(self.device)
            y_test = torch.from_numpy(np.array(y_test)).float().to(self.device)
            
            logger.info(f"📊 Test set size: {len(X_test)} sequences")
            
        except Exception as e:
            logger.error(f"Error preparing data for {symbol}: {e}")
            return None
        
        # Make predictions
        with torch.no_grad():
            # New model predictions
            try:
                new_pred = new_model(X_test)
                if new_pred.dim() > 2:
                    new_pred = new_pred[:, -1, :]
                new_pred_np = new_pred.cpu().numpy()
                logger.info(f"✅ New model predictions: shape {new_pred_np.shape}")
            except Exception as e:
                logger.error(f"Error with new model predictions: {e}")
                return None
            
            # Old model predictions (if available)
            old_pred_np = None
            if old_model is not None:
                try:
                    old_pred = old_model(X_test)
                    if old_pred.dim() > 2:
                        old_pred = old_pred[:, -1, :]
                    old_pred_np = old_pred.cpu().numpy()
                    logger.info(f"✅ Old model predictions: shape {old_pred_np.shape}")
                except Exception as e:
                    logger.warning(f"⚠️  Error with old model predictions: {e}")
        
        # Calculate metrics
        y_test_np = y_test.cpu().numpy()
        
        new_metrics = self.calculate_metrics(y_test_np, new_pred_np)
        logger.info(f"\n📊 New Model Metrics ({Path(new_model_path).stem}):")
        logger.info(f"   MAE:  {new_metrics['mae']:.6f}")
        logger.info(f"   RMSE: {new_metrics['rmse']:.6f}")
        logger.info(f"   R²:   {new_metrics['r2']:.4f}")
        logger.info(f"   MAPE: {new_metrics['mape']:.2f}%")
        
        result = {
            'symbol': symbol,
            'test_samples': len(X_test),
            'new_model_path': new_model_path,
            'new_metrics': new_metrics,
            'old_model_path': old_model_path,
            'old_metrics': None,
            'improvement': None
        }
        
        if old_pred_np is not None:
            old_metrics = self.calculate_metrics(y_test_np, old_pred_np)
            logger.info(f"\n📊 Old Model Metrics ({Path(old_model_path).stem}):")
            logger.info(f"   MAE:  {old_metrics['mae']:.6f}")
            logger.info(f"   RMSE: {old_metrics['rmse']:.6f}")
            logger.info(f"   R²:   {old_metrics['r2']:.4f}")
            logger.info(f"   MAPE: {old_metrics['mape']:.2f}%")
            
            result['old_metrics'] = old_metrics
            
            # Calculate improvements
            improvements = {
                'mae_improvement': ((old_metrics['mae'] - new_metrics['mae']) / old_metrics['mae'] * 100) if old_metrics['mae'] != 0 else 0,
                'rmse_improvement': ((old_metrics['rmse'] - new_metrics['rmse']) / old_metrics['rmse'] * 100) if old_metrics['rmse'] != 0 else 0,
                'r2_improvement': new_metrics['r2'] - old_metrics['r2'],
                'mape_improvement': ((old_metrics['mape'] - new_metrics['mape']) / old_metrics['mape'] * 100) if old_metrics['mape'] != np.inf else 0
            }
            
            result['improvement'] = improvements
            
            logger.info(f"\n📈 Improvement (New vs Old):")
            logger.info(f"   MAE:  {improvements['mae_improvement']:+.2f}%")
            logger.info(f"   RMSE: {improvements['rmse_improvement']:+.2f}%")
            logger.info(f"   R²:   {improvements['r2_improvement']:+.4f}")
            logger.info(f"   MAPE: {improvements['mape_improvement']:+.2f}%")
        
        return result
    
    def compare_all(self, symbols=None):
        """Compare all available models.
        
        Args:
            symbols: List of symbols to compare. If None, auto-detect.
        """
        if symbols is None:
            symbols = ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOGE', 'MATIC', 'AVAX', 'LINK']
        
        new_models_dir = Path('models/saved_models')
        old_models_dir = Path('models/saved_models_old')
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🤖 Model Comparison: New vs Old")
        logger.info(f"{'='*70}")
        logger.info(f"New models dir: {new_models_dir}")
        logger.info(f"Old models dir: {old_models_dir if old_models_dir.exists() else 'NOT FOUND'}")
        
        for symbol in symbols:
            new_model_path = new_models_dir / f"{symbol}_lstm_model.pth"
            old_model_path = old_models_dir / f"{symbol}_lstm_model.pth" if old_models_dir.exists() else None
            
            if not new_model_path.exists():
                logger.warning(f"⚠️  New model not found: {new_model_path}")
                continue
            
            result = self.test_symbol(symbol, str(new_model_path), str(old_model_path) if old_model_path else None)
            if result:
                self.results.append(result)
    
    def generate_report(self, output_file='model_comparison_report.txt'):
        """Generate comparison report.
        
        Args:
            output_file: Output file name
        """
        if not self.results:
            logger.warning("⚠️  No results to report")
            return
        
        report = []
        report.append("="*80)
        report.append(f"🤖 Model Comparison Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Device: {self.device}")
        report.append("="*80)
        report.append("")
        
        # Summary table
        report.append("📊 SUMMARY")
        report.append("-" * 80)
        report.append(f"{'Symbol':<10} {'Samples':<10} {'New MAE':<12} {'Old MAE':<12} {'Improvement':<12}")
        report.append("-" * 80)
        
        for result in self.results:
            symbol = result['symbol']
            samples = result['test_samples']
            new_mae = result['new_metrics']['mae']
            
            if result['old_metrics']:
                old_mae = result['old_metrics']['mae']
                improvement = f"{result['improvement']['mae_improvement']:+.2f}%"
            else:
                old_mae = "N/A"
                improvement = "N/A"
            
            report.append(f"{symbol:<10} {samples:<10} {new_mae:<12.6f} {str(old_mae):<12} {improvement:<12}")
        
        report.append("")
        report.append("="*80)
        report.append("📈 DETAILED RESULTS")
        report.append("="*80)
        report.append("")
        
        for result in self.results:
            symbol = result['symbol']
            report.append(f"\n{symbol} CRYPTOCURRENCY")
            report.append("-" * 80)
            report.append(f"Test Samples: {result['test_samples']}")
            report.append("")
            
            # New model metrics
            report.append("New Model Metrics:")
            for key, value in result['new_metrics'].items():
                if key == 'mape' and value != np.inf:
                    report.append(f"  {key.upper()}: {value:.2f}%")
                else:
                    report.append(f"  {key.upper()}: {value:.6f}")
            
            # Old model metrics and comparison
            if result['old_metrics']:
                report.append("")
                report.append("Old Model Metrics:")
                for key, value in result['old_metrics'].items():
                    if key == 'mape' and value != np.inf:
                        report.append(f"  {key.upper()}: {value:.2f}%")
                    else:
                        report.append(f"  {key.upper()}: {value:.6f}")
                
                report.append("")
                report.append("Improvement (New vs Old):")
                for key, value in result['improvement'].items():
                    if 'improvement' in key:
                        if 'r2' in key:
                            report.append(f"  {key}: {value:+.4f}")
                        else:
                            report.append(f"  {key}: {value:+.2f}%")
            
            report.append("")
        
        # Overall statistics
        report.append("\n" + "="*80)
        report.append("📊 OVERALL STATISTICS")
        report.append("="*80)
        report.append("")
        
        new_maes = [r['new_metrics']['mae'] for r in self.results]
        report.append(f"Average New Model MAE: {np.mean(new_maes):.6f}")
        
        if any(r['old_metrics'] for r in self.results):
            old_maes = [r['old_metrics']['mae'] for r in self.results if r['old_metrics']]
            improvements = [r['improvement']['mae_improvement'] for r in self.results if r['improvement']]
            
            report.append(f"Average Old Model MAE: {np.mean(old_maes):.6f}")
            report.append(f"Average Improvement: {np.mean(improvements):+.2f}%")
        
        report.append("")
        report.append("="*80)
        
        # Write to file
        report_text = "\n".join(report)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"\n✅ Report saved to: {output_file}")
        print(report_text)  # Also print to console


def main():
    """Main function."""
    try:
        # Initialize comparator
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        comparator = ModelComparator(device=device)
        
        # Compare all models
        comparator.compare_all()
        
        # Generate report
        comparator.generate_report('model_comparison_report.txt')
        
        logger.info("\n✅ Comparison complete!")
        return True
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
