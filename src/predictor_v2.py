#!/usr/bin/env python3
"""
🎯 Advanced Predictor Module V2

- Multi-step ahead predictions (3-5 candles)
- Ensemble predictions (weighted averaging)
- Confidence intervals
- Real-time price streaming
- Signal generation for trading
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
from src.model_tft import TemporalFusionTransformer
from src.data_fetcher_tft import TFTDataFetcher

logger = logging.getLogger(__name__)


class PredictorV2:
    """Advanced multi-step prediction engine"""
    
    def __init__(self, model_path: str, symbol: str = 'SOL', device: str = 'cuda'):
        """
        Args:
            model_path: Path to trained model
            symbol: Cryptocurrency symbol
            device: Device to use (cuda/cpu)
        """
        self.model_path = model_path
        self.symbol = symbol
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.fetcher = TFTDataFetcher()
        
        # Load model
        self.model = None
        self.scaler = None
        self._load_model()
        
        logger.info(f"🎯 PredictorV2 initialized for {symbol}")
    
    def _load_model(self):
        """Load trained TFT model"""
        try:
            # Create dummy model to get input size
            dummy_model = TemporalFusionTransformer(
                input_size=8,  # Default
                hidden_size=256,
                num_heads=8,
                num_layers=2,
                dropout=0.2
            )
            
            state_dict = torch.load(self.model_path, map_location=self.device)
            dummy_model.load_state_dict(state_dict)
            self.model = dummy_model.to(self.device)
            self.model.eval()
            
            logger.info(f"✓ Model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"✗ Failed to load model: {e}")
            raise
    
    def predict_single_step(self, X: np.ndarray) -> np.ndarray:
        """Predict next price (single step)
        
        Args:
            X: Feature sequence (lookback, features)
        
        Returns:
            Predicted price
        """
        X_tensor = torch.tensor(X[np.newaxis, :, :], dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            pred = self.model(X_tensor).cpu().numpy().flatten()
        
        return pred
    
    def predict_multi_step(self, X: np.ndarray, steps: int = 5,
                          confidence: float = 0.95) -> Dict:
        """Predict multiple steps ahead with confidence intervals
        
        Args:
            X: Latest feature sequence
            steps: Number of steps to predict
            confidence: Confidence level for intervals (0.95 = 95%)
        
        Returns:
            Dictionary with predictions and confidence intervals
        """
        predictions = []
        uncertainties = []
        
        current_X = X.copy()
        
        with torch.no_grad():
            for step in range(steps):
                # Make prediction
                X_tensor = torch.tensor(current_X[np.newaxis, :, :], 
                                       dtype=torch.float32).to(self.device)
                pred = self.model(X_tensor).cpu().numpy().flatten()[0]
                predictions.append(pred)
                
                # Estimate uncertainty (increases with prediction horizon)
                uncertainty = 0.1 * (step + 1) / steps  # Proportional to horizon
                uncertainties.append(uncertainty)
                
                # Update sequence for next prediction
                current_X = np.roll(current_X, -1, axis=0)
                current_X[-1, 0] = pred  # Update last close price
        
        # Calculate confidence intervals
        predictions = np.array(predictions)
        uncertainties = np.array(uncertainties)
        z_score = 1.96 if confidence == 0.95 else 1.64  # 95% or 90%
        
        lower_bounds = predictions - z_score * uncertainties
        upper_bounds = predictions + z_score * uncertainties
        
        return {
            'predictions': predictions,
            'lower_bounds': lower_bounds,
            'upper_bounds': upper_bounds,
            'uncertainties': uncertainties,
            'confidence_level': confidence
        }
    
    def ensemble_predict(self, X: np.ndarray, steps: int = 5,
                        num_samples: int = 10) -> Dict:
        """Ensemble prediction with Monte Carlo dropout
        
        Args:
            X: Feature sequence
            steps: Prediction steps
            num_samples: Number of Monte Carlo samples
        
        Returns:
            Ensemble predictions with statistics
        """
        all_predictions = []
        
        # Enable dropout for uncertainty estimation
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()
        
        for _ in range(num_samples):
            preds = self.predict_multi_step(X, steps=steps)
            all_predictions.append(preds['predictions'])
        
        all_predictions = np.array(all_predictions)
        
        # Calculate statistics
        ensemble_mean = np.mean(all_predictions, axis=0)
        ensemble_std = np.std(all_predictions, axis=0)
        ensemble_lower = np.percentile(all_predictions, 2.5, axis=0)
        ensemble_upper = np.percentile(all_predictions, 97.5, axis=0)
        
        return {
            'mean': ensemble_mean,
            'std': ensemble_std,
            'lower_bound_95': ensemble_lower,
            'upper_bound_95': ensemble_upper,
            'all_samples': all_predictions
        }
    
    def predict_with_actual_sequence(self, df: pd.DataFrame, lookback: int = 60,
                                    steps: int = 5) -> Dict:
        """Make predictions with real market data
        
        Args:
            df: OHLCV dataframe
            lookback: Lookback window
            steps: Prediction steps
        
        Returns:
            Predictions with current price and forecast
        """
        # Add indicators
        df = self.fetcher.add_tft_indicators(df)
        
        # Prepare features
        X, y, scaler = self.fetcher.prepare_ml_features(df, lookback=lookback)
        
        if X is None:
            logger.error("Failed to prepare features")
            return None
        
        # Get latest sequence
        X_latest = X[-1]
        current_price = y[-1]
        
        # Make predictions
        preds_dict = self.predict_multi_step(X_latest, steps=steps)
        preds_raw = preds_dict['predictions']
        
        # Inverse transform predictions
        num_features = X.shape[2]
        preds_full = np.zeros((len(preds_raw), num_features))
        preds_full[:, 0] = preds_raw
        preds_inverse = scaler.inverse_transform(preds_full)[:, 0]
        
        # Calculate metrics
        price_changes = ((preds_inverse - current_price) / current_price) * 100
        
        return {
            'current_price': current_price,
            'predictions': preds_inverse,
            'price_changes': price_changes,
            'timestamp': datetime.now(),
            'lookback': lookback,
            'steps': steps,
            'confidence_bounds': {
                'lower': preds_dict['lower_bounds'],
                'upper': preds_dict['upper_bounds']
            }
        }
    
    def generate_trading_signals(self, predictions: Dict) -> Dict:
        """Generate trading signals from predictions
        
        Args:
            predictions: Prediction results from predict_with_actual_sequence
        
        Returns:
            Trading signals and recommendations
        """
        if predictions is None:
            return None
        
        current_price = predictions['current_price']
        preds = predictions['predictions']
        
        # Calculate trend strength
        trend = np.mean(np.diff(preds))
        trend_direction = 'UP' if trend > 0 else 'DOWN' if trend < 0 else 'NEUTRAL'
        
        # Volatility of predictions
        volatility = np.std(preds)
        
        # Support and resistance levels
        support = np.min(preds)
        resistance = np.max(preds)
        
        # Signal strength (0-100)
        trend_strength = abs(trend) / (volatility + 1e-6) * 100
        trend_strength = min(trend_strength, 100)
        
        # Generate signal
        if trend > volatility * 0.5 and trend_strength > 50:
            signal = 'BUY'
            signal_strength = min(trend_strength, 100)
        elif trend < -volatility * 0.5 and trend_strength > 50:
            signal = 'SELL'
            signal_strength = min(trend_strength, 100)
        else:
            signal = 'HOLD'
            signal_strength = 50
        
        return {
            'signal': signal,
            'signal_strength': signal_strength,
            'trend': trend_direction,
            'trend_magnitude': trend,
            'volatility': volatility,
            'support': support,
            'resistance': resistance,
            'entry_price': current_price,
            'take_profit': resistance,
            'stop_loss': support,
            'risk_reward_ratio': abs(resistance - current_price) / abs(current_price - support) if support != current_price else 0
        }
    
    def forecast_report(self, symbol: str, lookback: int = 60, steps: int = 5) -> str:
        """Generate comprehensive forecast report
        
        Args:
            symbol: Crypto symbol
            lookback: Lookback window
            steps: Prediction steps
        
        Returns:
            Formatted report string
        """
        try:
            # Fetch data
            trading_pair = f"{symbol}/USDT"
            df = self.fetcher.fetch_ohlcv_binance(trading_pair, timeframe='1h', limit=5000)
            
            if df is None:
                return f"Failed to fetch data for {symbol}"
            
            # Make predictions
            predictions = self.predict_with_actual_sequence(df, lookback=lookback, steps=steps)
            signals = self.generate_trading_signals(predictions)
            
            if predictions is None or signals is None:
                return "Failed to generate predictions"
            
            # Format report
            report = f"""
🎯 CRYPTO PRICE FORECAST REPORT
{'='*60}
📅 Report Time: {predictions['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
💵 Symbol: {symbol.upper()}/USDT
📈 Lookback: {lookback} hours | Forecast: {steps} candles

📊 CURRENT STATUS:
  Current Price: ${predictions['current_price']:.4f}
  24h Trend: {signals['trend']}
  Signal: {signals['signal']} (Strength: {signals['signal_strength']:.1f}%)

📈 FORECAST (Next {steps} Candles):
"""
            
            for i, (pred, change) in enumerate(zip(predictions['predictions'], 
                                                   predictions['price_changes']), 1):
                emoji = '📈' if change > 0 else '📉'
                report += f"  Candle +{i}: ${pred:.4f} ({change:+.2f}%) {emoji}\n"
            
            report += f"""
🔢 TECHNICAL LEVELS:
  Support: ${signals['support']:.4f}
  Resistance: ${signals['resistance']:.4f}
  Volatility: ${signals['volatility']:.4f}
  Risk/Reward: {signals['risk_reward_ratio']:.2f}:1

🎨 TRADING RECOMMENDATION:
  Action: {signals['signal']}
  Entry: ${signals['entry_price']:.4f}
  Take Profit: ${signals['take_profit']:.4f}
  Stop Loss: ${signals['stop_loss']:.4f}
  
{'='*60}
⚠️  Disclaimer: This is AI-generated prediction. Always DYOR.
"""
            
            return report
        
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return f"Error generating report: {e}"


if __name__ == "__main__":
    # Example usage
    predictor = PredictorV2(
        model_path='models/saved_models/SOL_tft_model.pth',
        symbol='SOL',
        device='cuda'
    )
    
    # Generate forecast report
    report = predictor.forecast_report('SOL', lookback=60, steps=5)
    print(report)
