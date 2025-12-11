"""Price prediction engine for cryptocurrencies."""

import logging
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from .model_trainer import LSTMModel, TransformerModel
from .data_fetcher import DataFetcher
from .technical_analysis import TechnicalAnalyzer

logger = logging.getLogger(__name__)


class Predictor:
    """Main predictor for cryptocurrency prices."""
    
    def __init__(self, model_path: str, model_type: str = 'lstm',
                 lookback: int = 60, prediction_horizon: int = 7,
                 config: Dict = None):
        """Initialize predictor.
        
        Args:
            model_path: Path to trained model
            model_type: Type of model (lstm or transformer)
            lookback: Number of days for lookback
            prediction_horizon: Days to predict ahead
            config: Configuration dictionary
        """
        self.model_type = model_type
        self.lookback = lookback
        self.prediction_horizon = prediction_horizon
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data_fetcher = DataFetcher(config)
        self.technical_analyzer = TechnicalAnalyzer()
        
        self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """Load trained model from disk.
        
        Args:
            model_path: Path to model file
        """
        try:
            if self.model_type == 'lstm':
                self.model = LSTMModel(
                    input_size=17,  # Number of features
                    hidden_size=128,
                    num_layers=2,
                    dropout=0.2
                )
            elif self.model_type == 'transformer':
                self.model = TransformerModel(
                    input_size=17,
                    d_model=512,
                    num_heads=8,
                    num_layers=4
                )
            
            if os.path.exists(model_path):
                self.model.load_state_dict(
                    torch.load(model_path, map_location=self.device)
                )
                logger.info(f"Model loaded from {model_path}")
            else:
                logger.warning(f"Model not found at {model_path}. Using untrained model.")
            
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    def predict_price(self, df: pd.DataFrame) -> Dict:
        """Predict future price for a cryptocurrency.
        
        Args:
            df: DataFrame with OHLCV and technical indicators
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Prepare features
            feature_cols = [
                'open', 'high', 'low', 'close', 'volume',
                'SMA_10', 'SMA_20', 'SMA_50',
                'RSI', 'MACD', 'MACD_diff',
                'BB_upper', 'BB_lower', 'ATR',
                'Volume_ratio', 'Daily_return', 'Price_momentum'
            ]
            
            df_clean = df[feature_cols].dropna().tail(self.lookback)
            
            if len(df_clean) < self.lookback:
                logger.warning(f"Insufficient data. Got {len(df_clean)}, need {self.lookback}")
                return None
            
            # Scale data
            scaled_data = self.scaler.fit_transform(df_clean)
            X = torch.from_numpy(scaled_data).float().unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                prediction = self.model(X).cpu().numpy()[0][0]
            
            current_price = df['close'].iloc[-1]
            predicted_price = prediction
            price_change = ((predicted_price - current_price) / current_price) * 100
            
            # Calculate confidence based on historical accuracy
            confidence = self._calculate_confidence(df)
            
            result = {
                'current_price': float(current_price),
                'predicted_price': float(predicted_price),
                'price_change_percent': float(price_change),
                'confidence': float(confidence),
                'prediction_date': df.index[-1] + timedelta(days=self.prediction_horizon),
                'timestamp': datetime.now()
            }
            
            logger.info(f"Prediction - Current: {current_price:.2f}, "
                       f"Predicted: {predicted_price:.2f}, "
                       f"Change: {price_change:.2f}%, "
                       f"Confidence: {confidence:.2f}%")
            return result
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None
    
    def predict_path(self, df: pd.DataFrame, steps: int = None) -> List[Dict]:
        """Predict future price path (multi-step prediction).
        
        Args:
            df: DataFrame with OHLCV data
            steps: Number of steps to predict (defaults to prediction_horizon)
            
        Returns:
            List of prediction dictionaries for each step
        """
        try:
            steps = steps or self.prediction_horizon
            predictions = []
            current_df = df.copy()
            
            for i in range(steps):
                pred = self.predict_price(current_df)
                if pred is None:
                    break
                
                predictions.append(pred)
                
                # Create synthetic row for next iteration
                new_row = current_df.iloc[-1].copy()
                new_row['close'] = pred['predicted_price']
                current_df = pd.concat([current_df, pd.DataFrame([new_row])])
            
            logger.info(f"Generated {len(predictions)}-step price path")
            return predictions
        except Exception as e:
            logger.error(f"Failed to generate price path: {e}")
            return None
    
    def find_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Find support and resistance levels.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with support/resistance levels
        """
        try:
            levels = self.technical_analyzer.find_support_resistance(df)
            return levels
        except Exception as e:
            logger.error(f"Failed to find S/R levels: {e}")
            return None
    
    def generate_trading_signal(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Generate comprehensive trading signal.
        
        Args:
            symbol: Cryptocurrency symbol
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with trading signal and recommendations
        """
        try:
            # Get predictions
            prediction = self.predict_price(df)
            if prediction is None:
                return None
            
            # Get S/R levels
            sr_levels = self.find_support_resistance(df)
            
            # Get technical signals
            tech_signals = self.technical_analyzer.get_signals(df)
            
            current_price = df['close'].iloc[-1]
            predicted_price = prediction['predicted_price']
            confidence = prediction['confidence']
            
            # Calculate entry/exit points
            signal = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'current_price': float(current_price),
                'predicted_price': float(predicted_price),
                'predicted_change': float(prediction['price_change_percent']),
                'confidence': float(confidence),
                'support_level': float(sr_levels.get('support', current_price * 0.95)),
                'resistance_level': float(sr_levels.get('resistance', current_price * 1.05)),
                'entry_min': float(sr_levels.get('support', current_price * 0.98)),
                'entry_max': float(sr_levels.get('support', current_price * 1.00)),
                'take_profit': [
                    float(current_price * 1.03),
                    float(current_price * 1.06),
                    float(current_price * 1.10)
                ],
                'stop_loss': float(sr_levels.get('support', current_price * 0.95)),
                'rsi': float(tech_signals.get('rsi', 50)),
                'macd': float(tech_signals.get('macd_diff', 0)),
                'trend': tech_signals.get('trend', 'NEUTRAL'),
                'recommendation': self._generate_recommendation(prediction, tech_signals)
            }
            
            logger.info(f"Trading signal generated for {symbol}: {signal['recommendation']}")
            return signal
        except Exception as e:
            logger.error(f"Failed to generate trading signal: {e}")
            return None
    
    def _calculate_confidence(self, df: pd.DataFrame) -> float:
        """Calculate prediction confidence score.
        
        Args:
            df: DataFrame with historical data
            
        Returns:
            Confidence score between 0 and 100
        """
        try:
            # Calculate based on volatility and trend strength
            returns = df['close'].pct_change()
            volatility = returns.std()
            
            # Lower volatility = higher confidence
            confidence = max(0, min(100, 100 - (volatility * 500)))
            return confidence
        except:
            return 65.0  # Default confidence
    
    def _generate_recommendation(self, prediction: Dict, tech_signals: Dict) -> str:
        """Generate trading recommendation.
        
        Args:
            prediction: Price prediction
            tech_signals: Technical analysis signals
            
        Returns:
            Recommendation string
        """
        change = prediction['price_change_percent']
        confidence = prediction['confidence']
        trend = tech_signals.get('trend', 'NEUTRAL')
        
        if confidence < 50:
            return "WAIT"
        elif change > 5 and trend == "UPTREND":
            return "BUY"
        elif change < -5 and trend == "DOWNTREND":
            return "SELL"
        elif change > 2:
            return "BUY_WEAK"
        elif change < -2:
            return "SELL_WEAK"
        else:
            return "HOLD"
