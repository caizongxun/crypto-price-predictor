"""Technical analysis module for cryptocurrency trading signals."""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TechnicalAnalyzer:
    """Performs technical analysis on cryptocurrency data."""
    
    def __init__(self, config: Dict = None):
        """Initialize technical analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.sr_periods = self.config.get('support_resistance_periods', 20)
        self.rsi_period = self.config.get('rsi_period', 14)
        self.macd_fast = self.config.get('macd_fast', 12)
        self.macd_slow = self.config.get('macd_slow', 26)
        self.macd_signal = self.config.get('macd_signal', 9)
    
    def find_support_resistance(self, df: pd.DataFrame, period: int = None) -> Dict:
        """Find support and resistance levels.
        
        Args:
            df: DataFrame with OHLCV data
            period: Period for finding extrema (defaults to config)
            
        Returns:
            Dictionary with support and resistance levels
        """
        try:
            period = period or self.sr_periods
            
            # Find local minima (support) and maxima (resistance)
            high = df['high'].tail(period)
            low = df['low'].tail(period)
            
            support = low.min()
            resistance = high.max()
            
            # Find multiple levels
            highs = high.nlargest(3).tolist()
            lows = low.nsmallest(3).tolist()
            
            result = {
                'support': float(support),
                'resistance': float(resistance),
                'support_levels': [float(x) for x in lows],
                'resistance_levels': [float(x) for x in highs],
                'midpoint': float((support + resistance) / 2)
            }
            
            logger.info(f"S/R - Support: {support:.2f}, Resistance: {resistance:.2f}")
            return result
        except Exception as e:
            logger.error(f"Failed to find S/R levels: {e}")
            return {}
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        """Calculate Relative Strength Index.
        
        Args:
            df: DataFrame with close prices
            period: RSI period (defaults to config)
            
        Returns:
            Series with RSI values
        """
        try:
            period = period or self.rsi_period
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            logger.error(f"Failed to calculate RSI: {e}")
            return pd.Series()
    
    def calculate_macd(self, df: pd.DataFrame) -> Dict:
        """Calculate MACD indicator.
        
        Args:
            df: DataFrame with close prices
            
        Returns:
            Dictionary with MACD, signal line, and histogram
        """
        try:
            fast = df['close'].ewm(span=self.macd_fast).mean()
            slow = df['close'].ewm(span=self.macd_slow).mean()
            macd = fast - slow
            signal = macd.ewm(span=self.macd_signal).mean()
            histogram = macd - signal
            
            return {
                'macd': macd.iloc[-1],
                'signal': signal.iloc[-1],
                'histogram': histogram.iloc[-1],
                'trend': 'BULLISH' if histogram.iloc[-1] > 0 else 'BEARISH'
            }
        except Exception as e:
            logger.error(f"Failed to calculate MACD: {e}")
            return {}
    
    def detect_divergence(self, df: pd.DataFrame) -> Dict:
        """Detect RSI divergence patterns.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with divergence information
        """
        try:
            rsi = self.calculate_rsi(df)
            close = df['close']
            
            if len(rsi) < 2 or len(close) < 2:
                return {}
            
            # Check for bullish divergence (lower lows in price, higher lows in RSI)
            bullish_div = (
                close.iloc[-1] < close.iloc[-2] and
                rsi.iloc[-1] > rsi.iloc[-2] and
                rsi.iloc[-1] < 30
            )
            
            # Check for bearish divergence (higher highs in price, lower highs in RSI)
            bearish_div = (
                close.iloc[-1] > close.iloc[-2] and
                rsi.iloc[-1] < rsi.iloc[-2] and
                rsi.iloc[-1] > 70
            )
            
            return {
                'bullish_divergence': bullish_div,
                'bearish_divergence': bearish_div,
                'rsi': rsi.iloc[-1],
                'price_change': (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100
            }
        except Exception as e:
            logger.error(f"Failed to detect divergence: {e}")
            return {}
    
    def detect_breakout(self, df: pd.DataFrame, period: int = 20) -> Dict:
        """Detect support/resistance breakouts.
        
        Args:
            df: DataFrame with OHLCV data
            period: Period for breakout detection
            
        Returns:
            Dictionary with breakout information
        """
        try:
            high = df['high'].tail(period).max()
            low = df['low'].tail(period).min()
            current_price = df['close'].iloc[-1]
            
            breakout_up = current_price > high
            breakout_down = current_price < low
            
            return {
                'breakout_up': breakout_up,
                'breakout_down': breakout_down,
                'high_level': float(high),
                'low_level': float(low),
                'current_price': float(current_price),
                'strength': abs((current_price - (high + low) / 2) / ((high - low) / 2)) if high != low else 0
            }
        except Exception as e:
            logger.error(f"Failed to detect breakout: {e}")
            return {}
    
    def calculate_volatility(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate historical volatility.
        
        Args:
            df: DataFrame with close prices
            period: Period for volatility calculation
            
        Returns:
            Volatility as percentage
        """
        try:
            returns = df['close'].pct_change().tail(period)
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized
            return float(volatility)
        except Exception as e:
            logger.error(f"Failed to calculate volatility: {e}")
            return 0.0
    
    def get_signals(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive technical signals.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with all technical signals
        """
        try:
            rsi = self.calculate_rsi(df)
            macd = self.calculate_macd(df)
            divergence = self.detect_divergence(df)
            breakout = self.detect_breakout(df)
            volatility = self.calculate_volatility(df)
            
            # Determine trend
            if macd.get('histogram', 0) > 0 and rsi.iloc[-1] > 50:
                trend = 'UPTREND'
            elif macd.get('histogram', 0) < 0 and rsi.iloc[-1] < 50:
                trend = 'DOWNTREND'
            else:
                trend = 'NEUTRAL'
            
            # Count bullish/bearish signals
            signals = 0
            if rsi.iloc[-1] < 30:
                signals += 1  # Oversold
            elif rsi.iloc[-1] > 70:
                signals -= 1  # Overbought
            
            if divergence.get('bullish_divergence', False):
                signals += 2
            elif divergence.get('bearish_divergence', False):
                signals -= 2
            
            if breakout.get('breakout_up', False):
                signals += 1
            elif breakout.get('breakout_down', False):
                signals -= 1
            
            return {
                'rsi': float(rsi.iloc[-1]),
                'rsi_signal': 'oversold' if rsi.iloc[-1] < 30 else 'overbought' if rsi.iloc[-1] > 70 else 'neutral',
                'macd': float(macd.get('macd', 0)),
                'macd_signal': float(macd.get('signal', 0)),
                'macd_diff': float(macd.get('histogram', 0)),
                'macd_trend': macd.get('trend', 'NEUTRAL'),
                'divergence': divergence,
                'breakout': breakout,
                'volatility': volatility,
                'trend': trend,
                'signal_strength': signals,
                'overall_signal': 'BUY' if signals > 2 else 'SELL' if signals < -2 else 'HOLD'
            }
        except Exception as e:
            logger.error(f"Failed to generate signals: {e}")
            return {}
    
    def identify_patterns(self, df: pd.DataFrame) -> List[str]:
        """Identify candlestick patterns.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            List of detected patterns
        """
        patterns = []
        
        try:
            if len(df) < 3:
                return patterns
            
            close = df['close']
            open_price = df['open']
            high = df['high']
            low = df['low']
            
            # Hammer
            if (
                (high.iloc[-1] - max(open_price.iloc[-1], close.iloc[-1])) > (close.iloc[-1] - open_price.iloc[-1]) and
                (min(open_price.iloc[-1], close.iloc[-1]) - low.iloc[-1]) < (high.iloc[-1] - max(open_price.iloc[-1], close.iloc[-1]))
            ):
                patterns.append('HAMMER')
            
            # Engulfing
            if (
                close.iloc[-1] > open_price.iloc[-1] and
                open_price.iloc[-1] < close.iloc[-2] and
                close.iloc[-1] > open_price.iloc[-2]
            ):
                patterns.append('BULLISH_ENGULFING')
            
            # Doji
            if abs(close.iloc[-1] - open_price.iloc[-1]) < (high.iloc[-1] - low.iloc[-1]) * 0.1:
                patterns.append('DOJI')
            
            logger.info(f"Detected patterns: {patterns}")
            return patterns
        except Exception as e:
            logger.error(f"Failed to identify patterns: {e}")
            return patterns
