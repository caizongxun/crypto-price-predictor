import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class MultiTimeframeAnalyzer:
    """Analyze market trends across multiple timeframes (1h, 4h, 1d)"""
    
    def __init__(self):
        self.timeframes = ['1h', '4h', '1d']
    
    def analyze_structure(self, prices_1h: np.ndarray, prices_4h: np.ndarray, prices_1d: np.ndarray) -> Dict:
        """Analyze market structure across all timeframes"""
        
        analysis = {
            '1h': self._analyze_single_timeframe(prices_1h, '1h'),
            '4h': self._analyze_single_timeframe(prices_4h, '4h'),
            '1d': self._analyze_single_timeframe(prices_1d, '1d'),
            'alignment': 'NEUTRAL'
        }
        
        # Determine overall alignment
        trends = [analysis['1h']['trend'], analysis['4h']['trend'], analysis['1d']['trend']]
        if all(t == 'BULLISH' for t in trends):
            analysis['alignment'] = 'STRONG_BULLISH'
        elif all(t == 'BEARISH' for t in trends):
            analysis['alignment'] = 'STRONG_BEARISH'
        elif trends.count('BULLISH') >= 2:
            analysis['alignment'] = 'BULLISH'
        elif trends.count('BEARISH') >= 2:
            analysis['alignment'] = 'BEARISH'
            
        return analysis
    
    def _analyze_single_timeframe(self, prices: np.ndarray, timeframe: str) -> Dict:
        """Analyze trend and strength for a single timeframe"""
        try:
            prices = np.array(prices, dtype=float).flatten()
            if len(prices) < 20:
                return {'trend': 'NEUTRAL', 'confidence': 0, 'reason': 'Insufficient data'}
            
            # Calculate basic indicators
            current_price = prices[-1]
            sma_short = np.mean(prices[-5:])
            sma_medium = np.mean(prices[-20:])
            
            # Determine trend
            if sma_short > sma_medium:
                trend = 'BULLISH'
                strength = (sma_short - sma_medium) / sma_medium * 100
            else:
                trend = 'BEARISH' 
                strength = (sma_medium - sma_short) / sma_medium * 100
            
            # Calculate RSI
            delta = np.diff(prices)
            gain = (delta.where(delta > 0, 0)).mean() if hasattr(delta, 'where') else np.mean(delta[delta > 0]) if len(delta[delta > 0]) > 0 else 0
            loss = (-delta.where(delta < 0, 0)).mean() if hasattr(delta, 'where') else np.mean(-delta[delta < 0]) if len(delta[delta < 0]) > 0 else 0
            
            rs = gain / loss if loss != 0 else 0
            rsi = 100 - (100 / (1 + rs)) if loss != 0 else 50
            
            confidence = min(abs(strength) * 50 + (10 if trend == 'BULLISH' and rsi < 70 else 0), 100)
            
            return {
                'trend': trend,
                'confidence': float(confidence),
                'rsi': float(rsi),
                'sma_gap': float(strength)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing timeframe {timeframe}: {e}")
            return {'trend': 'NEUTRAL', 'confidence': 0}
