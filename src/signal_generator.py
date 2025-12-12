import numpy as np
import pandas as pd
import torch
import logging
from datetime import datetime
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class SignalType(Enum):
    STRONG_BUY = "🟢 STRONG BUY"
    BUY = "🟢 BUY"
    NEUTRAL = "⚪ NEUTRAL"
    SELL = "🔴 SELL"
    STRONG_SELL = "🔴 STRONG SELL"

class TrendDirection(Enum):
    STRONG_UPTREND = "📈 STRONG UPTREND"
    UPTREND = "📈 UPTREND"
    SIDEWAYS = "↔️ SIDEWAYS"
    DOWNTREND = "📉 DOWNTREND"
    STRONG_DOWNTREND = "📉 STRONG DOWNTREND"

@dataclass
class TradingSignal:
    symbol: str
    timestamp: datetime
    signal_type: SignalType
    current_price: float
    entry_price: float
    take_profit: float
    stop_loss: float
    confidence: float
    trend_direction: TrendDirection
    trend_strength: float
    predicted_prices: List[float] # Changed from predicted_next_price to list
    predicted_volatility: float
    momentum_score: float
    sentiment_score: float
    risk_reward_ratio: float
    is_breakout: bool
    technical_indicators: Dict

class SignalGenerator:
    def __init__(self, model=None, device='cpu'):
        self.model = model
        self.device = torch.device(device)
        self.lookback_period = 60
        self.prediction_steps = 5 # Predict 5 steps ahead
        self.min_confidence_threshold = 0.6
    
    def prepare_features(self, prices: np.ndarray) -> Optional[np.ndarray]:
        """用 17 個特徵準備資料供模型輸入"""
        try:
            prices = np.array(prices, dtype=float).flatten()
            
            if len(prices) < 60:
                logger.warning(f"Not enough price data: {len(prices)} < 60")
                return None
            
            # 創建 DataFrame 供計算技術指標
            df = pd.DataFrame({'close': prices})
            df['open'] = prices
            df['high'] = prices
            df['low'] = prices
            df['volume'] = np.ones_like(prices)
            
            # SMA
            df['SMA_10'] = df['close'].rolling(window=10).mean()
            df['SMA_20'] = df['close'].rolling(window=20).mean()
            df['SMA_50'] = df['close'].rolling(window=50).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            df['RSI'].fillna(50, inplace=True)
            
            # MACD
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            df['MACD'] = ema12 - ema26
            df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_diff'] = df['MACD'] - df['MACD_signal']
            
            # Bollinger Bands
            bb_mid = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['BB_upper'] = bb_mid + (bb_std * 2)
            df['BB_lower'] = bb_mid - (bb_std * 2)
            
            # ATR
            high_low = np.abs(np.diff(prices))
            df['ATR'] = pd.Series(np.concatenate([[0], high_low])).rolling(window=14).mean()
            df['ATR'].fillna(0, inplace=True)
            
            # Volume indicators
            df['Volume_SMA'] = df['volume'].rolling(window=20).mean()
            df['Volume_ratio'] = df['volume'] / (df['Volume_SMA'] + 1e-8)
            
            # Price change
            df['Daily_return'] = df['close'].pct_change()
            df['Price_momentum'] = df['close'].pct_change(periods=5)
            
            # 選擇 17 個特徵
            feature_cols = [
                'open', 'high', 'low', 'close', 'volume',
                'SMA_10', 'SMA_20', 'SMA_50',
                'RSI', 'MACD', 'MACD_diff',
                'BB_upper', 'BB_lower', 'ATR',
                'Volume_ratio', 'Daily_return', 'Price_momentum'
            ]
            
            # Forward fill 然後 backward fill 以處理 NaN
            features = df[feature_cols].ffill().bfill().fillna(0).values
            
            # 檢查是否有無效值
            if np.isnan(features).any() or np.isinf(features).any():
                logger.warning(f"Invalid values in features, cleaning...")
                features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
            
            # 正規化
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler(feature_range=(0, 1))
            features_normalized = scaler.fit_transform(features)
            
            return features_normalized
        
        except Exception as e:
            logger.error(f"❌ Error preparing features: {e}", exc_info=True)
            return None
    
    def predict_next_prices(self, prices: np.ndarray, symbol: str, current_price: float) -> Tuple[List[float], float]:
        """Predict next 5 prices and calculate volatility"""
        try:
            prices = np.array(prices, dtype=float).flatten()
            
            # 準備 17 個特徵
            features = self.prepare_features(prices)
            if features is None:
                raise ValueError(f"Failed to prepare features for {symbol}")
            
            if len(features) < self.lookback_period:
                raise ValueError(f"Not enough features: {len(features)} < {self.lookback_period}")
            
            # 取最後 60 個時間步
            X = features[-self.lookback_period:]
            X = X.reshape(1, X.shape[0], X.shape[1])  # (1, 60, 17)
            
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            
            was_training = self.model.training
            self.model.eval()
            
            with torch.no_grad():
                # Model output shape is now (1, 5)
                price_predictions_normalized = self.model(X_tensor).cpu().numpy()[0]
            
            if was_training:
                self.model.train()
            
            # 反正規化：將正規化的價格轉換回真實價格
            price_min = np.min(prices)
            price_max = np.max(prices)
            
            predicted_prices = []
            for pred_norm in price_predictions_normalized:
                pred_price = price_min + pred_norm * (price_max - price_min)
                # 簡單的範圍保護
                pred_price = np.clip(pred_price, price_min * 0.8, price_max * 1.2)
                predicted_prices.append(float(pred_price))
            
            # 計算波動率
            price_returns = np.diff(prices) / (prices[:-1] + 1e-8)
            predicted_volatility = float(np.std(price_returns) * np.sqrt(252))
            
            # Log result (showing last prediction)
            last_pred = predicted_prices[-1]
            change_pct = (last_pred - current_price) / (current_price + 1e-8) * 100
            logger.info(f"✅ Prediction for {symbol}: 5 steps, Final: ${last_pred:.2f} ({change_pct:+.2f}%)")
            
            return predicted_prices, predicted_volatility
        
        except Exception as e:
            logger.warning(f"⚠️ Model prediction failed for {symbol}: {str(e)[:100]}")
            # Fallback: linear projection
            try:
                trend = (prices[-1] - prices[-5]) / 5
                fallback_prices = [current_price + trend * i for i in range(1, 6)]
                return fallback_prices, 0.02
            except:
                return [current_price] * 5, 0.02
    
    def calculate_technical_indicators(self, prices: np.ndarray) -> Dict:
        indicators = {}
        try:
            prices = np.array(prices, dtype=float).flatten()
            if len(prices) >= 14:
                delta = np.diff(prices)
                gains = np.where(delta > 0, delta, 0)
                losses = np.where(delta < 0, -delta, 0)
                avg_gain = np.mean(gains) if len(gains) > 0 else 0
                avg_loss = np.mean(losses) if len(losses) > 0 else 0
                if avg_loss != 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                else:
                    rsi = 100 if avg_gain > 0 else 50
                indicators['rsi'] = float(rsi)
            if len(prices) >= 20:
                sma = np.mean(prices[-20:])
                std = np.std(prices[-20:])
                indicators['bb_upper'] = float(sma + 2 * std)
                indicators['bb_lower'] = float(sma - 2 * std)
                indicators['bb_middle'] = float(sma)
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
        return indicators
    
    def identify_support_resistance(self, prices: np.ndarray, lookback: int = 20) -> Tuple[float, float]:
        prices = np.array(prices, dtype=float).flatten()
        recent_prices = prices[-lookback:]
        return float(np.min(recent_prices)), float(np.max(recent_prices))
    
    def calculate_momentum_score(self, prices: np.ndarray) -> float:
        try:
            prices = np.array(prices, dtype=float).flatten()
            if len(prices) < 20:
                return 0.0
            short_term = (prices[-1] - prices[-5]) / (prices[-5] + 1e-8)
            long_term = (prices[-1] - prices[-20]) / (prices[-20] + 1e-8)
            roc = short_term * 0.6 + long_term * 0.4
            return float(np.clip(roc / 0.05, -1, 1))
        except:
            return 0.0
    
    def identify_trend(self, prices: np.ndarray, current_price: float) -> Tuple[TrendDirection, float]:
        try:
            prices = np.array(prices, dtype=float).flatten()
            if len(prices) < 60:
                return TrendDirection.SIDEWAYS, 0.0
            sma_short = float(np.mean(prices[-5:]))
            sma_medium = float(np.mean(prices[-20:]))
            sma_long = float(np.mean(prices[-60:]))
            trend_strength = abs(current_price - sma_medium) / (sma_medium + 1e-8)
            trend_strength = float(min(trend_strength, 1.0))
            if sma_short > sma_medium > sma_long:
                direction = TrendDirection.STRONG_UPTREND if trend_strength > 0.03 else TrendDirection.UPTREND
            elif sma_short < sma_medium < sma_long:
                direction = TrendDirection.STRONG_DOWNTREND if trend_strength > 0.03 else TrendDirection.DOWNTREND
            else:
                direction = TrendDirection.SIDEWAYS
            return direction, trend_strength
        except:
            return TrendDirection.SIDEWAYS, 0.0
    
    def detect_breakout(self, prices: np.ndarray, current_price: float) -> bool:
        try:
            prices = np.array(prices, dtype=float).flatten()
            if len(prices) < 20:
                return False
            recent_high = float(np.max(prices[-20:-1]))
            recent_low = float(np.min(prices[-20:-1]))
            threshold = 0.002
            return current_price > recent_high * (1 + threshold) or current_price < recent_low * (1 - threshold)
        except:
            return False
    
    def generate_signal(self, symbol: str, current_price: float, price_history: np.ndarray, volume_history: Optional[np.ndarray] = None) -> Optional[TradingSignal]:
        try:
            price_history = np.array(price_history, dtype=float).flatten()
            if len(price_history) < self.lookback_period:
                return None
            
            predicted_prices = []
            if self.model is not None:
                try:
                    predicted_prices, predicted_volatility = self.predict_next_prices(price_history[-self.lookback_period:], symbol, current_price)
                except Exception as e:
                    logger.warning(f"Model prediction error for {symbol}: {e}")
                    predicted_prices = [float(current_price)] * 5
                    predicted_volatility = 0.02
            else:
                predicted_prices = [float(current_price)] * 5
                predicted_volatility = 0.02
            
            # Get the final predicted price (5th step) for main logic
            final_predicted_price = predicted_prices[-1]
            
            technical_indicators = self.calculate_technical_indicators(price_history)
            support, resistance = self.identify_support_resistance(price_history)
            momentum_score = self.calculate_momentum_score(price_history)
            trend_direction, trend_strength = self.identify_trend(price_history, current_price)
            is_breakout = self.detect_breakout(price_history, current_price)
            rsi = technical_indicators.get('rsi', 50.0)
            
            signal_type, confidence = self._generate_signal_type(current_price, final_predicted_price, rsi, momentum_score, trend_strength, trend_direction, is_breakout, technical_indicators)
            entry_price, take_profit, stop_loss = self._calculate_entry_exit_points(current_price, final_predicted_price, support, resistance, trend_direction, signal_type, predicted_volatility)
            
            if signal_type in [SignalType.STRONG_BUY, SignalType.BUY]:
                risk = entry_price - stop_loss
                reward = take_profit - entry_price
            elif signal_type in [SignalType.STRONG_SELL, SignalType.SELL]:
                risk = stop_loss - entry_price
                reward = entry_price - take_profit
            else:
                risk = abs(stop_loss - entry_price)
                reward = abs(take_profit - entry_price)
            
            risk_reward_ratio = float(reward / (risk + 1e-8) if risk != 0 else 0)
            sentiment_score = float(momentum_score * 0.3 + (trend_strength if trend_direction in [TrendDirection.STRONG_UPTREND, TrendDirection.UPTREND] else -trend_strength) * 0.4 + (1 if is_breakout else 0) * 0.3)
            
            return TradingSignal(symbol=symbol, timestamp=datetime.now(), signal_type=signal_type, current_price=float(current_price), entry_price=float(entry_price), take_profit=float(take_profit), stop_loss=float(stop_loss), confidence=float(confidence), trend_direction=trend_direction, trend_strength=float(trend_strength), predicted_prices=predicted_prices, predicted_volatility=float(predicted_volatility), momentum_score=float(momentum_score), sentiment_score=float(sentiment_score), risk_reward_ratio=float(risk_reward_ratio), is_breakout=bool(is_breakout), technical_indicators=technical_indicators)
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}", exc_info=True)
            return None
    
    def _generate_signal_type(self, current_price: float, predicted_price: float, rsi: float, momentum_score: float, trend_strength: float, trend_direction: TrendDirection, is_breakout: bool, technical_indicators: Dict) -> Tuple[SignalType, float]:
        confidence = 0.5
        signals = []
        
        # 模型預測是最重要的
        if current_price > 0:
            price_change = (predicted_price - current_price) / (current_price + 1e-8)
            model_signal = 1.0 if price_change > 0.005 else (-1.0 if price_change < -0.005 else 0.0)
            signals.append(model_signal)
            if abs(price_change) > 0.005:
                confidence += 0.25
        
        # RSI 信號
        rsi_signal = 1.0 if rsi < 30 else (-1.0 if rsi > 70 else 0.0)
        signals.append(rsi_signal)
        if rsi < 30 or rsi > 70:
            confidence += 0.10
        
        # Momentum 信號
        signals.append(momentum_score)
        confidence += abs(momentum_score) * 0.08
        
        # 趨勢 信號
        trend_signal = (trend_strength if trend_direction in [TrendDirection.STRONG_UPTREND, TrendDirection.UPTREND] else (-trend_strength if trend_direction in [TrendDirection.STRONG_DOWNTREND, TrendDirection.DOWNTREND] else 0.0))
        signals.append(trend_signal)
        confidence += abs(trend_signal) * 0.10
        
        # Breakout 信號
        if is_breakout:
            breakout_signal = 1.0 if trend_direction in [TrendDirection.UPTREND, TrendDirection.STRONG_UPTREND] else -1.0
            signals.append(breakout_signal)
            confidence += 0.15
        
        overall_signal = float(np.mean(signals)) if signals else 0.0
        confidence = float(min(confidence, 0.95))
        
        if overall_signal > 0.3:
            return (SignalType.STRONG_BUY, confidence) if confidence > 0.75 else (SignalType.BUY, confidence)
        elif overall_signal < -0.3:
            return (SignalType.STRONG_SELL, confidence) if confidence > 0.75 else (SignalType.SELL, confidence)
        return SignalType.NEUTRAL, 0.5
    
    def _calculate_entry_exit_points(self, current_price: float, predicted_price: float, support: float, resistance: float, trend_direction: TrendDirection, signal_type: SignalType, predicted_volatility: float) -> Tuple[float, float, float]:
        volatility_factor = max(float(predicted_volatility), 0.01)
        
        if signal_type in [SignalType.STRONG_BUY, SignalType.BUY]:
            entry_price = float(current_price)
            take_profit = float(max(predicted_price, resistance))
            stop_loss = float(support)
        elif signal_type in [SignalType.STRONG_SELL, SignalType.SELL]:
            entry_price = float(current_price)
            take_profit = float(min(predicted_price, support))
            stop_loss = float(resistance)
        else:
            entry_price = float(current_price)
            take_profit = float(current_price * (1 + volatility_factor))
            stop_loss = float(current_price * (1 - volatility_factor))
        
        return entry_price, take_profit, stop_loss
