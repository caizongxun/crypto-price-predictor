import asyncio
import logging
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
import requests
import torch
import torch.nn as nn

from src.signal_generator import SignalGenerator, TradingSignal
from src.discord_bot_handler import DiscordBotHandler

load_dotenv()
logger = logging.getLogger(__name__)


# ===== 定義模型架構（與訓練時完全相同）=====

class EnhancedLSTMModel(nn.Module):
    """Enhanced LSTM with batch norm and better regularization"""
    def __init__(self, input_size: int, hidden_size: int = 256, num_layers: int = 4):
        super(EnhancedLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.input_bn = nn.BatchNorm1d(input_size)
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers,
            batch_first=True,
            dropout=0.4 if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.attention = nn.MultiheadAttention(
            hidden_size * 2,
            num_heads=16,
            dropout=0.3,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(-1, x.shape[-1])
        x = self.input_bn(x)
        x = x.view(batch_size, -1, x.shape[-1])
        
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        last_out = attn_out[:, -1, :]
        
        output = self.fc(last_out)
        return output


class GRUModel(nn.Module):
    """Enhanced GRU with batch norm"""
    def __init__(self, input_size: int, hidden_size: int = 256, num_layers: int = 4):
        super(GRUModel, self).__init__()
        
        self.input_bn = nn.BatchNorm1d(input_size)
        
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.4 if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(-1, x.shape[-1])
        x = self.input_bn(x)
        x = x.view(batch_size, -1, x.shape[-1])
        
        gru_out, _ = self.gru(x)
        gru_out = self.layer_norm(gru_out)
        
        last_out = gru_out[:, -1, :]
        output = self.fc(last_out)
        return output


class TransformerEncoderModel(nn.Module):
    """Transformer-based model for better sequence learning"""
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 3):
        super(TransformerEncoderModel, self).__init__()
        
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.positional_encoding = nn.Parameter(torch.randn(1, 60, hidden_size))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=512,
            dropout=0.3,
            batch_first=True,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        x = self.input_projection(x)
        x = x + self.positional_encoding[:, :x.shape[1], :]
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        output = self.fc(x)
        return output


class EnsembleModel(nn.Module):
    """Advanced ensemble - fusion of 3 models (LSTM + GRU + Transformer)"""
    def __init__(self, lstm_model, gru_model, transformer_model):
        super(EnsembleModel, self).__init__()
        self.lstm_model = lstm_model
        self.gru_model = gru_model
        self.transformer_model = transformer_model
        
        self.fusion = nn.Sequential(
            nn.Linear(3, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        lstm_out = self.lstm_model(x)
        gru_out = self.gru_model(x)
        transformer_out = self.transformer_model(x)
        
        combined = torch.cat([lstm_out, gru_out, transformer_out], dim=1)
        output = self.fusion(combined)
        
        return output


# ===== 交易機器人 =====

class RealtimeTradingBot:
    """實時交易信號機器人 - 使用融合模型"""
    
    def __init__(self, device: str = 'cpu'):
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')
        self.device = torch.device(device)
        
        # 初始化 Binance US Client
        self.client = None
        try:
            from binance.client import Client
            
            self.client = Client(
                self.api_key, 
                self.api_secret,
                tld='us'
            )
            
            self.client.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            })
            
            self.client.ping()
            logger.info("✅ Binance US Client initialized successfully!")
        
        except Exception as e:
            logger.warning(f"⚠️ Binance US failed: {str(e)[:80]}")
            self.client = None
        
        # 初始化 Discord Handler
        self.discord_handler = DiscordBotHandler()
        self.discord_thread = threading.Thread(target=self.discord_handler.start, daemon=True)
        self.discord_thread.start()
        logger.info("🤖 Discord Bot started in background thread")
        
        # 監控配置
        self.symbols = ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOGE', 'MATIC', 'AVAX', 'LINK']
        self.interval = '1h'
        self.lookback_period = 60
        self.check_frequency = 900  # 15 分鐘
        
        # 為每個幣種加載模型
        self.models = {}
        self._load_models()
        
        # 為每個幣種創建信號生成器
        self.signal_generators = {}
        for symbol in self.symbols:
            model = self.models.get(symbol)
            logger.info(f"🔧 SignalGenerator for {symbol}: model={'✅ Loaded' if model is not None else '❌ None'}, device={device}")
            self.signal_generators[symbol] = SignalGenerator(model=model, device=device)
        
        logger.info("📊 Signal Generators initialized for all symbols with ensemble models")
        
        # 信號歷史
        self.signal_history: Dict[str, TradingSignal] = {}
        self.last_signal_time: Dict[str, datetime] = {}
        
        logger.info("✅ RealtimeTradingBot initialized")
        logger.info(f"⏱️  Check frequency: {self.check_frequency // 60} minutes")
    
    def _load_models(self):
        """為每個幣種加載已訓練的融合模型"""
        model_dir = "models/saved_models"
        
        for symbol in self.symbols:
            model_path = f"{model_dir}/{symbol}_lstm_model.pth"
            
            try:
                if os.path.exists(model_path):
                    logger.info(f"Loading model from {model_path}")
                    
                    # 加載 state_dict
                    state_dict = torch.load(model_path, map_location=self.device)
                    
                    # 檢查 state_dict 的結構
                    first_key = list(state_dict.keys())[0]
                    logger.info(f"First key in state_dict: {first_key}")
                    
                    # 判斷是否是包裝的 ensemble 模型
                    if first_key.startswith('lstm_model.'):
                        logger.info(f"Detected wrapped ensemble model for {symbol}")
                        
                        # 直接創建並加載 ensemble
                        lstm_model = EnhancedLSTMModel(input_size=17, hidden_size=256, num_layers=4)
                        gru_model = GRUModel(input_size=17, hidden_size=256, num_layers=4)
                        transformer_model = TransformerEncoderModel(input_size=17, hidden_size=128, num_layers=3)
                        ensemble = EnsembleModel(lstm_model, gru_model, transformer_model)
                        
                        # 使用 strict=False 加載
                        ensemble.load_state_dict(state_dict, strict=False)
                        ensemble.eval()
                        ensemble.to(self.device)
                        
                        self.models[symbol] = ensemble
                        logger.info(f"✅ Loaded wrapped ensemble model for {symbol}")
                    else:
                        logger.warning(f"⚠️ Unknown state_dict format for {symbol}, first key: {first_key}")
                        self.models[symbol] = None
                else:
                    logger.warning(f"⚠️ Model not found for {symbol}")
                    self.models[symbol] = None
            
            except Exception as e:
                logger.error(f"❌ Error loading model for {symbol}: {str(e)[:200]}")
                self.models[symbol] = None
    
    def fetch_klines_binance_us(self, symbol: str, interval: str = '1h', limit: int = 100) -> Optional[List]:
        """從 Binance US 獲取 K 線數據"""
        try:
            if not self.client:
                return None
            
            binance_symbol = f"{symbol}USDT"
            klines = self.client.get_klines(
                symbol=binance_symbol,
                interval=interval,
                limit=limit
            )
            return klines
        except Exception as e:
            logger.warning(f"Binance US fetch failed for {symbol}: {str(e)[:80]}")
            return None
    
    def fetch_klines_from_coingecko(self, symbol: str, days: int = 60) -> Optional[List]:
        """使用 CoinGecko 作為備用"""
        try:
            coingecko_id = {
                'BTC': 'bitcoin',
                'ETH': 'ethereum',
                'BNB': 'binancecoin',
                'XRP': 'ripple',
                'ADA': 'cardano',
                'SOL': 'solana',
                'DOGE': 'dogecoin',
                'MATIC': 'matic-network',
                'AVAX': 'avalanche-2',
                'LINK': 'chainlink'
            }.get(symbol, symbol.lower())
            
            url = f"https://api.coingecko.com/api/v3/coins/{coingecko_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'daily'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            return data.get('prices', [])
        
        except Exception as e:
            logger.error(f"CoinGecko error for {symbol}: {str(e)[:80]}")
            return None
    
    def parse_klines_to_prices(self, klines: List) -> np.ndarray:
        """解析 K 線數據為價格數組"""
        if isinstance(klines[0], (list, tuple)):
            prices = np.array([float(k[4]) for k in klines])
        else:
            prices = np.array([float(k[1]) for k in klines])
        
        return prices
    
    def _send_signal_notification_sync(self, symbol: str, signal: TradingSignal):
        """通過 Discord 發送信號通知 (同步版本)"""
        try:
            import discord
            
            if "BUY" in signal.signal_type.value:
                color = discord.Color.green()
            elif "SELL" in signal.signal_type.value:
                color = discord.Color.red()
            else:
                color = discord.Color.yellow()
            
            embed = discord.Embed(
                title=f"{signal.signal_type.value}",
                description=f"**{symbol}USDT** 交易信號",
                color=color,
                timestamp=datetime.now()
            )
            
            embed.add_field(name="💰 當前價格", value=f"${signal.current_price:,.2f}", inline=True)
            embed.add_field(name="🎯 進場價", value=f"${signal.entry_price:,.2f}", inline=True)
            embed.add_field(name="📊 信心度", value=f"{signal.confidence:.2%}", inline=True)
            
            embed.add_field(name="✅ 獲利目標", value=f"${signal.take_profit:,.2f}", inline=True)
            embed.add_field(name="❌ 止損點", value=f"${signal.stop_loss:,.2f}", inline=True)
            embed.add_field(name="⚖️ 風險回報比", value=f"{signal.risk_reward_ratio:.2f}", inline=True)
            
            embed.add_field(name="📈 趨勢", value=signal.trend_direction.value, inline=True)
            embed.add_field(name="💪 趨勢強度", value=f"{signal.trend_strength:.2%}", inline=True)
            embed.add_field(name="🔥 是否突破", value="✅ 是" if signal.is_breakout else "❌ 否", inline=True)
            
            embed.add_field(name="⚠️ 免責聲明", value="此信號僅供參考，請自行評估風險後決定交易。", inline=False)
            
            embed.set_footer(text="Crypto Price Predictor Bot")
            
            # 使用 discord_handler 的隊列發送，不直接使用 async/await
            self.discord_handler.queue_embed(embed)
            logger.info(f"✅ Signal queued for Discord for {symbol}")
        
        except Exception as e:
            logger.error(f"Error queuing signal notification: {e}")
    
    def process_symbol(self, symbol: str) -> Optional[TradingSignal]:
        """處理單個交易對並生成交易信號"""
        try:
            # 獲取 K 線數據
            klines = self.fetch_klines_binance_us(symbol)
            
            if not klines:
                prices = self.fetch_klines_from_coingecko(symbol)
                if not prices:
                    logger.warning(f"❌ Could not fetch data for {symbol}")
                    return None
                data_source = "CoinGecko"
            else:
                prices = self.parse_klines_to_prices(klines)
                data_source = "Binance US"
            
            logger.info(f"✅ Processing {symbol}USDT ({data_source}) - {len(prices)} data points")
            
            if len(prices) < self.lookback_period:
                logger.warning(f"⚠️ {symbol}: Insufficient data")
                return None
            
            current_price = float(prices[-1])
            
            # 使用對應幣種的信號生成器（帶有模型）
            signal_gen = self.signal_generators.get(symbol)
            
            logger.debug(f"🔧 Calling generate_signal for {symbol}, model={'✅' if signal_gen.model else '❌'}")
            
            signal = signal_gen.generate_signal(
                symbol=symbol,
                current_price=current_price,
                price_history=prices
            )
            
            if signal:
                logger.info(f"📈 Signal generated for {symbol}: {signal.signal_type.value} (Confidence: {signal.confidence:.2%})")
                
                if self._should_send_signal(symbol, signal):
                    self._send_signal_notification_sync(symbol, signal)
                    self.signal_history[symbol] = signal
                    self.last_signal_time[symbol] = datetime.now()
                
                return signal
            else:
                logger.info(f"⚪ No strong signal for {symbol}")
                return None
        
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}", exc_info=True)
            return None
    
    def _should_send_signal(self, symbol: str, signal: TradingSignal) -> bool:
        """判斷是否應該發送信號通知"""
        if symbol in self.last_signal_time:
            time_since_last = datetime.now() - self.last_signal_time[symbol]
            if time_since_last.total_seconds() < 3600:
                return False
        
        if signal.confidence < 0.50:
            return False
        
        return True
    
    def run_monitoring_loop(self):
        """運行持續監控循環"""
        logger.info("🚀 Starting real-time trading bot monitoring...")
        logger.info("📢 Discord Bot 通知已啓用")
        logger.info(f"⏱️  檢查頻率: 每 15 分鐘一次")
        
        import time
        time.sleep(2)
        
        while True:
            try:
                logger.info(f"\n{'='*70}")
                logger.info(f"Scanning {len(self.symbols)} symbols at {datetime.now()}")
                logger.info(f"{'='*70}")
                
                results = []
                for symbol in self.symbols:
                    result = self.process_symbol(symbol)
                    results.append(result)
                
                signals_generated = sum(1 for r in results if r is not None)
                strong_signals = sum(1 for r in results if r and r.confidence > 0.75)
                
                logger.info(f"📊 Generated {signals_generated} signals ({strong_signals} strong signals)")
                logger.info(f"⏰ Next check in {self.check_frequency // 60} minutes...")
                
                import time
                time.sleep(self.check_frequency)
            
            except KeyboardInterrupt:
                logger.info("⛔ Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                import time
                time.sleep(60)
    
    def start(self):
        """啟動機器人"""
        try:
            self.run_monitoring_loop()
        except Exception as e:
            logger.error(f"Fatal error: {e}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("="*70)
    logger.info("🤖 Crypto Price Predictor - Realtime Trading Bot")
    logger.info("="*70)
    
    bot = RealtimeTradingBot(device='cpu')
    bot.start()
