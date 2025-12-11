import numpy as np
import pandas as pd
import torch
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """交易信號類型"""
    STRONG_BUY = "🟢 STRONG BUY"
    BUY = "🟢 BUY"
    NEUTRAL = "⚪ NEUTRAL"
    SELL = "🔴 SELL"
    STRONG_SELL = "🔴 STRONG SELL"


class TrendDirection(Enum):
    """趨勢方向"""
    STRONG_UPTREND = "📈 STRONG UPTREND"
    UPTREND = "📈 UPTREND"
    SIDEWAYS = "↔️ SIDEWAYS"
    DOWNTREND = "📉 DOWNTREND"
    STRONG_DOWNTREND = "📉 STRONG DOWNTREND"


@dataclass
class TradingSignal:
    """交易信號結構"""
    symbol: str
    timestamp: datetime
    signal_type: SignalType
    current_price: float
    entry_price: float
    take_profit: float
    stop_loss: float
    confidence: float  # 0-1
    trend_direction: TrendDirection
    trend_strength: float  # 0-1
    predicted_next_price: float
    predicted_volatility: float
    momentum_score: float  # -1 to 1
    sentiment_score: float  # -1 to 1
    risk_reward_ratio: float
    is_breakout: bool
    technical_indicators: Dict


class SignalGenerator:
    """實時交易信號生成器"""
    
    def __init__(self, model=None, device='cuda'):
        self.model = model
        self.device = torch.device(device)
        self.lookback_period = 60
        self.min_confidence_threshold = 0.6
    
    def predict_next_price_and_volatility(
        self,
        X: np.ndarray,
        symbol: str
    ) -> Tuple[float, float]:
        """
        預測下一時間步的價格和波動率
        
        Returns:
            (predicted_price, predicted_volatility)
        """
        try:
            # 確保輸入格式正確
            if len(X.shape) == 2:
                X = X.reshape(1, X.shape[0], X.shape[1])
            
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                # 預測價格
                price_prediction = self.model(X_tensor)
                predicted_price = price_prediction.cpu().numpy()[0][0]
            
            # 計算波動率（基於歷史價格變化）
            price_returns = np.diff(X[0, :, 0]) / X[0, :-1, 0]
            predicted_volatility = float(np.std(price_returns) * np.sqrt(252))  # 年化波動率
            
            return float(predicted_price), float(predicted_volatility)
        
        except Exception as e:
            logger.error(f"Failed to predict price for {symbol}: {e}")
            return 0.0, 0.0
    
    def calculate_technical_indicators(self, prices: np.ndarray) -> Dict:
        """
        計算技術指標
        """
        indicators = {}
        
        try:
            # RSI (Relative Strength Index)
            if len(prices) >= 14:
                delta = np.diff(prices)
                gain = np.where(delta > 0, delta, 0).mean()
                loss = np.where(delta < 0, -delta, 0).mean()
                rs = gain / loss if loss != 0 else 0
                indicators['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD (Moving Average Convergence Divergence)
            if len(prices) >= 26:
                ema12 = self._calculate_ema(prices, 12)
                ema26 = self._calculate_ema(prices, 26)
                macd = ema12 - ema26
                signal = self._calculate_ema(np.array([macd] * len(prices)), 9)[-1]
                indicators['macd'] = float(macd[-1])
                indicators['macd_signal'] = float(signal)
            
            # Bollinger Bands
            if len(prices) >= 20:
                sma = np.mean(prices[-20:])
                std = np.std(prices[-20:])
                indicators['bb_upper'] = float(sma + 2 * std)
                indicators['bb_lower'] = float(sma - 2 * std)
                indicators['bb_middle'] = float(sma)
            
            # Stochastic Oscillator
            if len(prices) >= 14:
                low = np.min(prices[-14:])
                high = np.max(prices[-14:])
                current = prices[-1]
                k = 100 * (current - low) / (high - low) if high != low else 50
                indicators['stochastic_k'] = float(k)
            
            # ATR (Average True Range) - 波動性指標
            if len(prices) >= 14:
                atr = self._calculate_atr(prices[-14:])
                indicators['atr'] = float(atr)
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
        
        return indicators
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """計算指數移動平均線"""
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        multiplier = 2 / (period + 1)
        
        for i in range(1, len(prices)):
            ema[i] = prices[i] * multiplier + ema[i-1] * (1 - multiplier)
        
        return ema
    
    def _calculate_atr(self, prices: np.ndarray) -> float:
        """計算平均真實波幅"""
        tr = np.abs(np.diff(prices))
        return np.mean(tr)
    
    def identify_support_resistance(
        self,
        prices: np.ndarray,
        lookback: int = 20
    ) -> Tuple[float, float]:
        """
        識別支持位和阻力位
        """
        recent_prices = prices[-lookback:]
        support = np.min(recent_prices)
        resistance = np.max(recent_prices)
        
        return float(support), float(resistance)
    
    def calculate_momentum_score(
        self,
        prices: np.ndarray
    ) -> float:
        """
        計算動量分數 (-1 到 1)
        -1: 強烈看跌
         0: 中立
         1: 強烈看漲
        """
        try:
            # 計算短期和長期動量
            short_term = (prices[-1] - prices[-5]) / prices[-5]  # 5 期變化
            long_term = (prices[-1] - prices[-20]) / prices[-20]  # 20 期變化
            
            # 計算 ROC (Rate of Change)
            roc = short_term * 0.6 + long_term * 0.4
            
            # 限制在 -1 到 1 之間
            momentum = np.clip(roc / 0.05, -1, 1)
            
            return float(momentum)
        
        except Exception as e:
            logger.error(f"Error calculating momentum: {e}")
            return 0.0
    
    def identify_trend(
        self,
        prices: np.ndarray,
        current_price: float
    ) -> Tuple[TrendDirection, float]:
        """
        識別趨勢方向和強度
        """
        try:
            # 計算移動平均線
            sma_short = np.mean(prices[-5:])
            sma_medium = np.mean(prices[-20:])
            sma_long = np.mean(prices[-60:])
            
            # 計算趨勢強度（基於價格與 MA 的距離）
            trend_strength = abs(current_price - sma_medium) / sma_medium
            trend_strength = min(trend_strength, 1.0)  # 限制在 0-1
            
            # 判斷趨勢方向
            if sma_short > sma_medium > sma_long:
                if trend_strength > 0.03:
                    direction = TrendDirection.STRONG_UPTREND
                else:
                    direction = TrendDirection.UPTREND
            elif sma_short < sma_medium < sma_long:
                if trend_strength > 0.03:
                    direction = TrendDirection.STRONG_DOWNTREND
                else:
                    direction = TrendDirection.DOWNTREND
            else:
                direction = TrendDirection.SIDEWAYS
            
            return direction, float(trend_strength)
        
        except Exception as e:
            logger.error(f"Error identifying trend: {e}")
            return TrendDirection.SIDEWAYS, 0.0
    
    def detect_breakout(
        self,
        prices: np.ndarray,
        current_price: float
    ) -> bool:
        """
        檢測突破信號
        """
        try:
            # 檢查是否突破 20 期高點或低點
            recent_high = np.max(prices[-20:-1])
            recent_low = np.min(prices[-20:-1])
            
            breakout_threshold = 0.002  # 0.2% 突破
            
            # 上升突破
            if current_price > recent_high * (1 + breakout_threshold):
                return True
            
            # 下降突破
            if current_price < recent_low * (1 - breakout_threshold):
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Error detecting breakout: {e}")
            return False
    
    def generate_signal(
        self,
        symbol: str,
        current_price: float,
        price_history: np.ndarray,
        volume_history: Optional[np.ndarray] = None
    ) -> Optional[TradingSignal]:
        """
        生成完整的交易信號
        
        Args:
            symbol: 幣種符號 (e.g., 'BTC')
            current_price: 當前價格
            price_history: 歷史價格數組 (最後一個是最新的)
            volume_history: 歷史成交量 (可選)
        
        Returns:
            TradingSignal 對象或 None
        """
        try:
            # 確保有足夠的歷史數據
            if len(price_history) < self.lookback_period:
                logger.warning(f"Insufficient price history for {symbol}")
                return None
            
            # 預測下一時間步價格和波動率
            if self.model is not None:
                predicted_price, predicted_volatility = self.predict_next_price_and_volatility(
                    price_history[-self.lookback_period:],
                    symbol
                )
            else:
                # 如果沒有模型，使用簡單預測
                predicted_price = current_price
                predicted_volatility = np.std(np.diff(price_history) / price_history[:-1])
            
            # 計算技術指標
            technical_indicators = self.calculate_technical_indicators(price_history)
            
            # 識別支持位和阻力位
            support, resistance = self.identify_support_resistance(price_history)
            
            # 計算動量分數
            momentum_score = self.calculate_momentum_score(price_history)
            
            # 識別趨勢
            trend_direction, trend_strength = self.identify_trend(price_history, current_price)
            
            # 檢測突破
            is_breakout = self.detect_breakout(price_history, current_price)
            
            # 計算 RSI 用於信號確認
            rsi = technical_indicators.get('rsi', 50)
            
            # 生成信號
            signal_type, confidence = self._generate_signal_type(
                current_price=current_price,
                predicted_price=predicted_price,
                rsi=rsi,
                momentum_score=momentum_score,
                trend_strength=trend_strength,
                trend_direction=trend_direction,
                is_breakout=is_breakout,
                technical_indicators=technical_indicators
            )
            
            # 計算進場和止損點
            entry_price, take_profit, stop_loss = self._calculate_entry_exit_points(
                current_price=current_price,
                support=support,
                resistance=resistance,
                trend_direction=trend_direction,
                signal_type=signal_type,
                predicted_volatility=predicted_volatility
            )
            
            # 計算風險回報比
            if signal_type in [SignalType.STRONG_BUY, SignalType.BUY]:
                risk = entry_price - stop_loss
                reward = take_profit - entry_price
            else:
                risk = stop_loss - entry_price
                reward = entry_price - take_profit
            
            risk_reward_ratio = reward / risk if risk != 0 else 0
            
            # 情感分數 (基於多個因素)
            sentiment_score = float(
                momentum_score * 0.3 +
                (trend_strength if trend_direction in [TrendDirection.STRONG_UPTREND, TrendDirection.UPTREND] else -trend_strength) * 0.4 +
                (1 if is_breakout else 0) * 0.3
            )
            
            return TradingSignal(
                symbol=symbol,
                timestamp=datetime.now(),
                signal_type=signal_type,
                current_price=float(current_price),
                entry_price=float(entry_price),
                take_profit=float(take_profit),
                stop_loss=float(stop_loss),
                confidence=float(confidence),
                trend_direction=trend_direction,
                trend_strength=float(trend_strength),
                predicted_next_price=float(predicted_price),
                predicted_volatility=float(predicted_volatility),
                momentum_score=float(momentum_score),
                sentiment_score=float(sentiment_score),
                risk_reward_ratio=float(risk_reward_ratio),
                is_breakout=bool(is_breakout),
                technical_indicators=technical_indicators
            )
        
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    def _generate_signal_type(
        self,
        current_price: float,
        predicted_price: float,
        rsi: float,
        momentum_score: float,
        trend_strength: float,
        trend_direction: TrendDirection,
        is_breakout: bool,
        technical_indicators: Dict
    ) -> Tuple[SignalType, float]:
        """
        根據多個因素生成信號類型和置信度
        """
        confidence = 0.5  # 基礎置信度
        signals = []  # 累計信號分數
        
        # 1. 基於預測價格的信號
        price_change = (predicted_price - current_price) / current_price
        if price_change > 0.02:  # 預測上升 > 2%
            signals.append(1.0)
            confidence += 0.1
        elif price_change < -0.02:  # 預測下跌 > 2%
            signals.append(-1.0)
            confidence += 0.1
        else:
            signals.append(0.0)
        
        # 2. 基於 RSI 的信號
        if rsi < 30:  # 超賣
            signals.append(1.0)
            confidence += 0.15
        elif rsi > 70:  # 超買
            signals.append(-1.0)
            confidence += 0.15
        else:
            signals.append(0.0)
        
        # 3. 基於動量的信號
        signals.append(momentum_score)
        confidence += abs(momentum_score) * 0.1
        
        # 4. 基於趨勢的信號
        if trend_direction in [TrendDirection.STRONG_UPTREND, TrendDirection.UPTREND]:
            trend_signal = trend_strength
        elif trend_direction in [TrendDirection.STRONG_DOWNTREND, TrendDirection.DOWNTREND]:
            trend_signal = -trend_strength
        else:
            trend_signal = 0.0
        signals.append(trend_signal)
        confidence += abs(trend_signal) * 0.15
        
        # 5. 基於突破的信號
        if is_breakout:
            if trend_direction in [TrendDirection.UPTREND, TrendDirection.STRONG_UPTREND]:
                signals.append(1.0)
            else:
                signals.append(-1.0)
            confidence += 0.2
        
        # 計算綜合信號
        overall_signal = np.mean(signals)
        
        # 限制置信度
        confidence = min(confidence, 0.95)
        
        # 生成信號類型
        if overall_signal > 0.5:
            if confidence > 0.8:
                return SignalType.STRONG_BUY, confidence
            else:
                return SignalType.BUY, confidence
        elif overall_signal < -0.5:
            if confidence > 0.8:
                return SignalType.STRONG_SELL, confidence
            else:
                return SignalType.SELL, confidence
        else:
            return SignalType.NEUTRAL, 0.5
    
    def _calculate_entry_exit_points(
        self,
        current_price: float,
        support: float,
        resistance: float,
        trend_direction: TrendDirection,
        signal_type: SignalType,
        predicted_volatility: float
    ) -> Tuple[float, float, float]:
        """
        計算進場點、止盈點和止損點
        """
        # 使用波動率來調整距離
        volatility_factor = max(predicted_volatility, 0.01)
        
        if signal_type in [SignalType.STRONG_BUY, SignalType.BUY]:
            # 買入信號
            # 進場點：在支持位附近或當前價格稍低位置
            entry_price = min(current_price, support * 1.001)
            
            # 止損點：支持位下方
            stop_loss = support * (1 - volatility_factor * 2)
            
            # 止盈點：阻力位或更高
            take_profit = resistance * (1 + volatility_factor)
        
        else:
            # 賣出信號
            # 進場點：在阻力位附近或當前價格稍高位置
            entry_price = max(current_price, resistance * 0.999)
            
            # 止損點：阻力位上方
            stop_loss = resistance * (1 + volatility_factor * 2)
            
            # 止盈點：支持位或更低
            take_profit = support * (1 - volatility_factor)
        
        return float(entry_price), float(take_profit), float(stop_loss)
