import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv
import os

from src.signal_generator import SignalGenerator, TradingSignal
from src.discord_bot_handler import DiscordBotHandler

load_dotenv()
logger = logging.getLogger(__name__)


class RealtimeTradingBot:
    """實時交易信號機器人"""
    
    def __init__(
        self,
        model=None,
        api_key: str = None,
        api_secret: str = None,
        device: str = 'cuda'
    ):
        self.api_key = api_key or os.getenv('BINANCE_API_KEY')
        self.api_secret = api_secret or os.getenv('BINANCE_API_SECRET')
        self.client = Client(self.api_key, self.api_secret)
        
        self.signal_generator = SignalGenerator(model=model, device=device)
        self.discord_handler = DiscordBotHandler()  # 只保留 Discord
        
        # 監控配置
        self.symbols = ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOGE', 'MATIC', 'AVAX', 'LINK']
        self.interval = '1h'  # 1小時間間隔
        self.lookback_period = 60  # 60根K線
        self.check_frequency = 300  # 5分鐘檢查一次
        
        # 信號歷史（避免重複發送相同信號）
        self.signal_history: Dict[str, TradingSignal] = {}
        self.last_signal_time: Dict[str, datetime] = {}
    
    def fetch_klines(
        self,
        symbol: str,
        interval: str = '1h',
        limit: int = 100
    ) -> Optional[List]:
        """
        從 Binance 獲取 K 線數據
        """
        try:
            # 轉換符號格式 (BTC → BTCUSDT)
            binance_symbol = f"{symbol}USDT"
            
            klines = self.client.get_klines(
                symbol=binance_symbol,
                interval=interval,
                limit=limit
            )
            
            return klines
        
        except BinanceAPIException as e:
            logger.error(f"Binance API error for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching klines for {symbol}: {e}")
            return None
    
    def parse_klines(
        self,
        klines: List
    ) -> tuple:
        """
        解析 K 線數據為價格和成交量數組
        """
        if not klines:
            return None, None
        
        prices = np.array([float(k[4]) for k in klines])  # 收盤價
        volumes = np.array([float(k[7]) for k in klines])  # 成交量
        
        return prices, volumes
    
    async def process_symbol(
        self,
        symbol: str
    ) -> Optional[TradingSignal]:
        """
        處理單個交易對
        """
        logger.info(f"Processing {symbol}...")
        
        try:
            # 獲取 K 線數據
            klines = self.fetch_klines(
                symbol=symbol,
                interval=self.interval,
                limit=self.lookback_period
            )
            
            if not klines:
                logger.warning(f"No klines data for {symbol}")
                return None
            
            # 解析數據
            prices, volumes = self.parse_klines(klines)
            if prices is None:
                return None
            
            current_price = prices[-1]
            
            # 生成信號
            signal = self.signal_generator.generate_signal(
                symbol=symbol,
                current_price=current_price,
                price_history=prices,
                volume_history=volumes
            )
            
            if signal is None:
                return None
            
            # 檢查是否應該發送通知（避免重複）
            should_notify = self._should_notify(symbol, signal)
            
            if should_notify:
                # 發送通知（只有 Discord）
                await self._send_discord_signal(signal)
                
                # 更新歷史
                self.signal_history[symbol] = signal
                self.last_signal_time[symbol] = datetime.now()
            
            return signal
        
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            return None
    
    def _should_notify(self, symbol: str, signal: TradingSignal) -> bool:
        """
        判斷是否應該發送通知
        
        規則:
        1. 信號類型改變
        2. 信心度大幅提升 (> 0.1)
        3. 距離上次信號超過 1 小時
        """
        # 第一次信號
        if symbol not in self.signal_history:
            return True
        
        last_signal = self.signal_history[symbol]
        last_time = self.last_signal_time.get(symbol)
        
        # 信號類型改變
        if signal.signal_type != last_signal.signal_type:
            return True
        
        # 信心度大幅提升
        if signal.confidence - last_signal.confidence > 0.15:
            return True
        
        # 距離上次信號超過 1 小時
        if last_time and (datetime.now() - last_time).total_seconds() > 3600:
            return True
        
        return False
    
    async def _send_discord_signal(self, signal: TradingSignal):
        """
        發送 Discord 交易信號
        """
        try:
            # 根據信號類型選擇顏色
            from src.signal_generator import SignalType
            if signal.signal_type in [SignalType.STRONG_BUY, SignalType.BUY]:
                color = 3066993  # Green
            elif signal.signal_type in [SignalType.STRONG_SELL, SignalType.SELL]:
                color = 15158332  # Red
            else:
                color = 12370112  # Gray
            
            # 構建嵌入
            embed = {
                'title': f"{signal.signal_type.value}",
                'description': f"🎯 Trading Signal Generated for {signal.symbol}",
                'color': color,
                'fields': [
                    {'name': 'Symbol', 'value': signal.symbol, 'inline': True},
                    {'name': 'Signal', 'value': signal.signal_type.value, 'inline': True},
                    {'name': 'Confidence', 'value': f"{signal.confidence*100:.1f}%", 'inline': True},
                    {'name': 'Current Price', 'value': f"${signal.current_price:.2f}", 'inline': True},
                    {'name': 'Entry Price', 'value': f"${signal.entry_price:.2f}", 'inline': True},
                    {'name': 'Take Profit', 'value': f"${signal.take_profit:.2f}", 'inline': True},
                    {'name': 'Stop Loss', 'value': f"${signal.stop_loss:.2f}", 'inline': True},
                    {'name': 'Risk/Reward', 'value': f"{signal.risk_reward_ratio:.2f}", 'inline': True},
                    {'name': 'Trend', 'value': signal.trend_direction.value, 'inline': True},
                    {'name': 'Trend Strength', 'value': f"{signal.trend_strength*100:.1f}%", 'inline': True},
                    {'name': 'Predicted Next Price', 'value': f"${signal.predicted_next_price:.2f}", 'inline': True},
                    {'name': 'Predicted Volatility', 'value': f"{signal.predicted_volatility*100:.2f}%", 'inline': True},
                    {'name': 'Momentum', 'value': f"{signal.momentum_score:.2f}", 'inline': True},
                    {'name': 'Sentiment', 'value': f"{signal.sentiment_score:.2f}", 'inline': True},
                    {'name': 'Breakout', 'value': "✅ Yes" if signal.is_breakout else "❌ No", 'inline': True},
                    {'name': 'RSI', 'value': f"{signal.technical_indicators.get('rsi', 0):.1f}", 'inline': True},
                ],
                'timestamp': signal.timestamp.isoformat()
            }
            
            cog = self.discord_handler.bot.get_cog('TrainingNotificationCog')
            if cog:
                await cog.send_status_update(
                    title=f"🎯 {signal.symbol} Trading Signal",
                    description=signal.signal_type.value,
                    fields={
                        'Entry': f"${signal.entry_price:.2f}",
                        'TP': f"${signal.take_profit:.2f}",
                        'SL': f"${signal.stop_loss:.2f}",
                        'R/R': f"{signal.risk_reward_ratio:.2f}",
                        'Confidence': f"{signal.confidence*100:.1f}%",
                        'Trend': signal.trend_direction.value
                    }
                )
            
            logger.info(f"Discord signal sent for {signal.symbol}")
        
        except Exception as e:
            logger.error(f"Error sending Discord signal: {e}")
    
    async def run_monitoring_loop(self):
        """
        運行持續監控循環
        """
        logger.info("Starting real-time trading bot monitoring...")
        logger.info("📢 Discord Bot 通知已啓用")
        logger.info("❌ Email 通知已禁用")
        logger.info("❌ Telegram 通知已禁用")
        
        while True:
            try:
                logger.info(f"\n{'='*70}")
                logger.info(f"Scanning {len(self.symbols)} symbols at {datetime.now()}")
                logger.info(f"{'='*70}")
                
                # 處理所有交易對
                tasks = [self.process_symbol(symbol) for symbol in self.symbols]
                results = await asyncio.gather(*tasks)
                
                # 記錄生成的信號
                signals_generated = sum(1 for r in results if r is not None)
                logger.info(f"Generated {signals_generated} signals in this cycle")
                
                # 等待下一個檢查週期
                logger.info(f"Next check in {self.check_frequency} seconds...")
                await asyncio.sleep(self.check_frequency)
            
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # 出錯後等 1 分鐘再重試
    
    def start(self):
        """
        啟動機器人
        """
        try:
            asyncio.run(self.run_monitoring_loop())
        except Exception as e:
            logger.error(f"Fatal error: {e}")
