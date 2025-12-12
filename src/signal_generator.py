import numpy as np
import pandas as pd
import torch
import logging
from datetime import datetime
from typing import Dict, Tuple, Optional
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
    predicted_next_price: float
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
            
            logger.debug(f"✅ Features shape: {features.shape}, Last row non-zero: {np.any(features[-1] != 0)}")
            
            # 正規化
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler(feature_range=(0, 1))
            features_normalized = scaler.fit_transform(features)
            
            return features_normalized
        
        except Exception as e:
            logger.error(f"❌ Error preparing features: {e}", exc_info=True)
            return None
    
    def predict_next_price_and_volatility(self, prices: np.ndarray, symbol: str, current_price: float) -> Tuple[float, float]:
        try:
            prices = np.array(prices, dtype=float).flatten()
            logger.info(f"🤖 [Model Prediction] {symbol}: Input {len(prices)} prices, Current: ${current_price:.2f}")
            
            # 準備 17 個特徵
            features = self.prepare_features(prices)
            if features is None:
                raise ValueError(f"Failed to prepare features for {symbol}")
            
            if len(features) < self.lookback_period:
                raise ValueError(f"Not enough features: {len(features)} < {self.lookback_period}")
            
            # 取最後 60 個時間步
            X = features[-self.lookback_period:]
            X = X.reshape(1, X.shape[0], X.shape[1])  # (1, 60, 17)
            
            logger.debug(f"X shape for model: {X.shape}, Sample values: {X[0, 0, :3]}")
            
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            
            was_training = self.model.training
            self.model.eval()
            
            with torch.no_grad():
                price_prediction = self.model(X_tensor)
                predicted_price_normalized = price_prediction.cpu().numpy()[0][0]
            
            if was_training:
                self.model.train()
            
            # 反正規化：將正規化的價格轉換回真實價格
            # 模型訓練時正規化了 close 價格，所以我們需要反正規化
            price_min = np.min(prices)
            price_max = np.max(prices)
            predicted_price = price_min + predicted_price_normalized * (price_max - price_min)
            
            # 如果預測價格超出範圍，截斷到有效範圍
            predicted_price = np.clip(predicted_price, price_min * 0.8, price_max * 1.2)
            
            # 計算波動率
            price_returns = np.diff(prices) / (prices[:-1] + 1e-8)
            predicted_volatility = float(np.std(price_returns) * np.sqrt(252))
            
            # 計算價格變化百分比
            price_change_pct = (predicted_price - current_price) / (current_price + 1e-8) * 100
            
            logger.info(f"✅ Model prediction for {symbol}: ${predicted_price:.2f} | Change: {price_change_pct:+.2f}% (normalized: {predicted_price_normalized:.4f}, vol: {predicted_volatility:.4f})")
            return float(predicted_price), float(predicted_volatility)
        
        except Exception as e:
            logger.warning(f"⚠️ Model prediction failed for {symbol}: {str(e)[:100]}")
            try:
                prices = np.array(prices, dtype=float).flatten()
                price_returns = np.diff(prices) / (prices[:-1] + 1e-8)
                volatility = float(np.std(price_returns) * np.sqrt(252))
                logger.info(f"↩️ Fallback for {symbol}: current_price=${current_price:.2f}")
            except:
                volatility = 0.02
            return current_price, volatility
    
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
            
            if self.model is not None:
                try:
                    predicted_price, predicted_volatility = self.predict_next_price_and_volatility(price_history[-self.lookback_period:], symbol, current_price)
                except Exception as e:
                    logger.warning(f"Model prediction error for {symbol}: {e}")
                    predicted_price = float(current_price)
                    price_returns = np.diff(price_history) / (price_history[:-1] + 1e-8)
                    predicted_volatility = float(np.std(price_returns) * np.sqrt(252))
            else:
                predicted_price = float(current_price)
                price_returns = np.diff(price_history) / (price_history[:-1] + 1e-8)
                predicted_volatility = float(np.std(price_returns) * np.sqrt(252))
            
            technical_indicators = self.calculate_technical_indicators(price_history)
            support, resistance = self.identify_support_resistance(price_history)
            momentum_score = self.calculate_momentum_score(price_history)
            trend_direction, trend_strength = self.identify_trend(price_history, current_price)
            is_breakout = self.detect_breakout(price_history, current_price)
            rsi = technical_indicators.get('rsi', 50.0)
            
            signal_type, confidence = self._generate_signal_type(current_price, predicted_price, rsi, momentum_score, trend_strength, trend_direction, is_breakout, technical_indicators)
            entry_price, take_profit, stop_loss = self._calculate_entry_exit_points(current_price, predicted_price, support, resistance, trend_direction, signal_type, predicted_volatility)
            
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
            
            return TradingSignal(symbol=symbol, timestamp=datetime.now(), signal_type=signal_type, current_price=float(current_price), entry_price=float(entry_price), take_profit=float(take_profit), stop_loss=float(stop_loss), confidence=float(confidence), trend_direction=trend_direction, trend_strength=float(trend_strength), predicted_next_price=float(predicted_price), predicted_volatility=float(predicted_volatility), momentum_score=float(momentum_score), sentiment_score=float(sentiment_score), risk_reward_ratio=float(risk_reward_ratio), is_breakout=bool(is_breakout), technical_indicators=technical_indicators)
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}", exc_info=True)
            return None
    
    def _generate_signal_type(self, current_price: float, predicted_price: float, rsi: float, momentum_score: float, trend_strength: float, trend_direction: TrendDirection, is_breakout: bool, technical_indicators: Dict) -> Tuple[SignalType, float]:
        confidence = 0.5
        signals = []
        
        # 模型預測是最重要的 - 菜鳥模型中心
        if current_price > 0:
            price_change = (predicted_price - current_price) / (current_price + 1e-8)
            # 增加模型預測的權重 (訓練好的模型已經騎鑑了)
            model_signal = 1.0 if price_change > 0.005 else (-1.0 if price_change < -0.005 else 0.0)
            signals.append(model_signal)
            # 模型預測有推估信心增加
            if abs(price_change) > 0.005:
                confidence += 0.25  # 大幅度增加 (from 0.1)
            logger.debug(f"Model signal: {model_signal:.2f}, price_change: {price_change*100:.2f}%, confidence boost: +0.25")
        
        # RSI 信號 - 超買/超賣區間
        # BUY: RSI < 30 (超賣)
        # SELL: RSI > 70 (超買)
        rsi_signal = 1.0 if rsi < 30 else (-1.0 if rsi > 70 else 0.0)
        signals.append(rsi_signal)
        if rsi < 30 or rsi > 70:
            confidence += 0.10
            logger.debug(f"RSI signal: {rsi_signal:.2f} (RSI={rsi:.1f})")
        
        # Momentum 信號
        signals.append(momentum_score)
        confidence += abs(momentum_score) * 0.08
        
        # 趨勢 信號
        trend_signal = (trend_strength if trend_direction in [TrendDirection.STRONG_UPTREND, TrendDirection.UPTREND] else (-trend_strength if trend_direction in [TrendDirection.STRONG_DOWNTREND, TrendDirection.DOWNTREND] else 0.0))
        signals.append(trend_signal)
        confidence += abs(trend_signal) * 0.10
        logger.debug(f"Trend signal: {trend_signal:.2f}, direction: {trend_direction.value}")
        
        # Breakout 信號
        if is_breakout:
            breakout_signal = 1.0 if trend_direction in [TrendDirection.UPTREND, TrendDirection.STRONG_UPTREND] else -1.0
            signals.append(breakout_signal)
            confidence += 0.15
            logger.debug(f"Breakout signal: {breakout_signal:.2f}")
        
        overall_signal = float(np.mean(signals)) if signals else 0.0
        confidence = float(min(confidence, 0.95))
        
        logger.debug(f"Overall signal: {overall_signal:.2f}, confidence: {confidence:.2%}, signals: {[f'{s:.2f}' for s in signals]}")
        
        # BUY 信號邏輯 (positive signal)
        if overall_signal > 0.3:
            return (SignalType.STRONG_BUY, confidence) if confidence > 0.75 else (SignalType.BUY, confidence)
        # SELL 信號邏輯 (negative signal) - 對稱的邏輯
        elif overall_signal < -0.3:
            return (SignalType.STRONG_SELL, confidence) if confidence > 0.75 else (SignalType.SELL, confidence)
        return SignalType.NEUTRAL, 0.5
    
    def _calculate_entry_exit_points(self, current_price: float, predicted_price: float, support: float, resistance: float, trend_direction: TrendDirection, signal_type: SignalType, predicted_volatility: float) -> Tuple[float, float, float]:
        """
        計算進場、獲利、止損點
        
        對於 BUY:
            - Entry: 接近 current_price 或 support
            - TP: 高於 current_price（向上）
            - SL: 低於 current_price（向下保護）
        
        對於 SELL:
            - Entry: 接近 current_price 或 resistance
            - TP: 低於 current_price（向下）
            - SL: 高於 current_price（向上保護）
        """
        volatility_factor = max(float(predicted_volatility), 0.01)
        
        if signal_type in [SignalType.STRONG_BUY, SignalType.BUY]:
            # 買入信號
            entry_price = float(current_price)  # 使用當前價格作為進場點
            # TP: 向上 (使用預測價格或 resistance)
            take_profit = float(max(predicted_price, resistance))
            # SL: 向下 (使用 support)
            stop_loss = float(support)
            
        elif signal_type in [SignalType.STRONG_SELL, SignalType.SELL]:
            # 賣出信號
            entry_price = float(current_price)  # 使用當前價格作為進場點
            # TP: 向下 (使用預測價格或 support)
            take_profit = float(min(predicted_price, support))
            # SL: 向上 (使用 resistance)
            stop_loss = float(resistance)
            
        else:
            # NEUTRAL - 使用保守的設置
            entry_price = float(current_price)
            # TP 和 SL 基於波動率
            take_profit = float(current_price * (1 + volatility_factor))
            stop_loss = float(current_price * (1 - volatility_factor))
        
        logger.debug(f"Entry: ${entry_price:.2f}, TP: ${take_profit:.2f}, SL: ${stop_loss:.2f}")
        return entry_price, take_profit, stop_loss
