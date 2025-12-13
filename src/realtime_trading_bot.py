import os
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import threading
from typing import Dict, List

from src.data_fetcher import DataFetcher
from src.model_trainer import ModelTrainer, LSTMModel
from src.signal_generator import SignalGenerator, SignalType, TradingSignal
from src.discord_bot_handler import DiscordBotHandler
from src.gemini_signal_validator import GeminiSignalValidator, GeminiAnalysis
from src.multi_timeframe_analyzer import MultiTimeframeAnalyzer
from src.technical_analysis import TechnicalAnalyzer
from src.plotting import generate_prediction_chart
from src.huggingface_model_manager import HuggingFaceModelManager

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 加載環境變量
load_dotenv()

class RealtimeTradingBot:
    def __init__(self, use_huggingface: bool = True):
        """
        Initialize the bot.
        
        Args:
            use_huggingface: Whether to download models from HuggingFace (True) 
                           or use local saved models (False)
        """
        self.symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT',
            'SOLUSDT', 'DOGEUSDT', 'MATICUSDT', 'AVAXUSDT', 'LINKUSDT'
        ]
        self.timeframe = '1h'
        self.check_interval = 900  # 15 minutes
        self.use_huggingface = use_huggingface
        
        self.data_fetcher = DataFetcher()
        self.signal_generators = {}
        self.discord_bot = DiscordBotHandler()
        self.gemini_validator = GeminiSignalValidator(api_key=os.getenv('GROQ_API_KEY'))
        self.mtf_analyzer = MultiTimeframeAnalyzer(self.data_fetcher)
        self.technical_analyzer = TechnicalAnalyzer()
        
        # Initialize HuggingFace model manager if enabled
        if self.use_huggingface:
            hf_repo_id = os.getenv('HUGGINGFACE_REPO_ID', 'zongowo111/crypto_model')
            self.hf_manager = HuggingFaceModelManager(repo_id=hf_repo_id)
        else:
            self.hf_manager = None
        
        self.last_check_time = {}
        self.active_signals = {}
        
        # 啟動 Discord Bot 線程
        self.discord_bot.start()
        
        # 初始化模型和信號生成器
        self._initialize_models()
        
    def _initialize_models(self):
        """為每個交易對加載訓練好的模型 (支持 HuggingFace 和本地模型)"""
        logger.info(f"🔧 Model Source: {'HuggingFace Hub' if self.use_huggingface else 'Local Storage'}")
        
        for symbol in self.symbols:
            try:
                symbol_short = symbol.replace('USDT', '')
                
                # 情況 1：使用 HuggingFace 模型
                if self.use_huggingface:
                    logger.info(f"📥 Downloading {symbol_short} model from HuggingFace...")
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model = self.hf_manager.load_model_from_hf(
                        symbol=symbol_short,
                        device=device,
                        model_type='lstm'
                    )
                    
                    if model is None:
                        logger.warning(f"⚠️ Failed to load {symbol_short} from HF, falling back to local")
                        self.use_huggingface = False  # Fall back to local mode
                        model = self._load_local_model(symbol)
                        if model is None:
                            logger.error(f"❌ Cannot load {symbol_short} from any source!")
                            continue
                    else:
                        logger.info(f"✅ {symbol_short} model loaded from HuggingFace")
                
                # 情況 2：使用本地模型
                else:
                    model = self._load_local_model(symbol)
                    if model is None:
                        logger.error(f"❌ Cannot load {symbol_short} locally!")
                        continue
                
                # 封裝成統一接口供 SignalGenerator 使用
                class EnsembleModelWrapper:
                    def __init__(self, lstm_model, device):
                        self.lstm = lstm_model
                        self.device = device
                        self.training = False
                    
                    def eval(self):
                        self.lstm.eval()
                    
                    def train(self):
                        self.lstm.train()
                        
                    def __call__(self, x):
                        return self.lstm(x)

                wrapped_model = EnsembleModelWrapper(model, device)
                logger.info(f"✅ Loaded wrapped ensemble model for {symbol_short}")
                
                self.signal_generators[symbol] = SignalGenerator(
                    model=wrapped_model,
                    device=device
                )
                
                logger.info(f"🔧 SignalGenerator for {symbol_short}: model=✅ Loaded, device={device}")
                
            except Exception as e:
                logger.error(f"Error initializing model for {symbol}: {e}", exc_info=True)
        
        logger.info(f"📊 Signal Generators initialized for all symbols")

    def _load_local_model(self, symbol: str):
        """Load model from local storage."""
        try:
            import torch
            symbol_short = symbol.replace('USDT', '')
            model_path = f"models/saved_models/{symbol_short}_lstm_model.pth"
            
            if not os.path.exists(model_path):
                logger.warning(f"⚠️ No local model found at {model_path}")
                return None
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model_trainer = ModelTrainer(model_type='lstm', config={
                'hidden_size': 256,
                'num_layers': 3,
                'dropout': 0.3
            })
            
            logger.info(f"📂 Loading {symbol_short} from local storage...")
            model_trainer.load_model(model_path, input_size=17)
            
            # Verify output size
            if model_trainer.model.fc2.out_features != 5:
                logger.warning(f"⚠️ {symbol_short} has wrong output size, re-initializing")
                model_trainer.create_model(input_size=17)
            
            logger.info(f"✅ {symbol_short} loaded from local storage")
            return model_trainer.model
            
        except Exception as e:
            logger.error(f"Error loading local model for {symbol}: {e}")
            return None

    def run(self):
        """主循環"""
        logger.info(f"✅ RealtimeTradingBot initialized")
        logger.info(f"⏱️  Check frequency: {self.check_interval//60} minutes")
        logger.info("🚀 Starting real-time trading bot monitoring...")
        logger.info("📢 Discord Bot 通知已啓用")
        
        while True:
            try:
                now = datetime.now()
                logger.info(f"\n{'='*70}\nScanning {len(self.symbols)} symbols at {now}\n{'='*70}")
                
                for symbol in self.symbols:
                    self._process_symbol(symbol)
                
                # 更新 Discord Portfolio 狀態
                self.discord_bot.bot.loop.call_soon_threadsafe(
                    self.discord_bot.queue_embed,
                    self._create_portfolio_update()
                )
                
                logger.info(f"⏰ Next check in {self.check_interval//60} minutes...")
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                logger.info("🛑 Bot stopped by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
                time.sleep(60)

    def _process_symbol(self, symbol: str):
        """處理單個交易對"""
        try:
            # 1. 獲取數據
            df = self.data_fetcher.get_historical_data(symbol, self.timeframe, limit=100)
            if df is None or len(df) < 60:
                logger.warning(f"⚠️ Insufficient data for {symbol}")
                return

            current_price = df['close'].iloc[-1]
            price_history = df['close'].values
            logger.info(f"✅ Processing {symbol} - {len(df)} data points")
            
            # 2. 生成信號
            signal_generator = self.signal_generators.get(symbol)
            if not signal_generator:
                logger.error(f"❌ No signal generator for {symbol}")
                return

            signal = signal_generator.generate_signal(
                symbol=symbol.replace('USDT', ''),
                current_price=current_price,
                price_history=df['close'].values,
                volume_history=df['volume'].values
            )
            
            if not signal:
                logger.warning(f"⚠️ Failed to generate signal for {symbol}")
                return

            # 3. 多時間週期分析
            short_term = self.mtf_analyzer.analyze_timeframe(symbol, '1h')
            medium_term = self.mtf_analyzer.analyze_timeframe(symbol, '4h')
            long_term = self.mtf_analyzer.analyze_timeframe(symbol, '1d')
            
            # 4. AI 驗證
            logger.info(f"📈 Signal generated for {signal.symbol}: {signal.signal_type.value} (Confidence: {signal.confidence:.2%})")
            logger.info(f"🤖 Requesting Gemini validation for {signal.symbol}...")
            
            ai_analysis = self.gemini_validator.validate_signal(
                symbol=signal.symbol,
                signal_type=signal.signal_type.value,
                confidence=signal.confidence * 100,
                current_price=current_price,
                short_term_analysis=short_term,
                medium_term_analysis=medium_term,
                long_term_analysis=long_term,
                technical_indicators=signal.technical_indicators
            )
            
            if ai_analysis:
                logger.info(f"✨ Gemini Analysis: Valid={ai_analysis.is_valid}, Score={ai_analysis.validity_score}")
                
                if ai_analysis.validity_score < 40:
                    logger.info(f"🚫 Signal filtered by Gemini: score={ai_analysis.validity_score}")
                    self._update_global_signal_state(signal, ai_analysis, filtered=True)
                    return
                
                if ai_analysis.entry_price:
                    signal.entry_price = ai_analysis.entry_price
                if ai_analysis.stop_loss:
                    signal.stop_loss = ai_analysis.stop_loss
                if ai_analysis.take_profit:
                    signal.take_profit = ai_analysis.take_profit
            else:
                logger.warning("⚠️ Gemini analysis failed, proceeding with original signal")
                ai_analysis = GeminiAnalysis(
                    is_valid=True, validity_score=50, entry_price=current_price,
                    stop_loss=signal.stop_loss, take_profit=signal.take_profit,
                    risk_reward_ratio=1.0, reasoning="AI Unavailable",
                    market_condition="Unknown", confidence_adjustment=0
                )

            # 5. 發送通知
            self._send_discord_alert(signal, ai_analysis, price_history)
            
            # 6. 更新全局狀態
            self._update_global_signal_state(signal, ai_analysis)
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}", exc_info=True)

    def _send_discord_alert(self, signal: TradingSignal, ai_analysis: GeminiAnalysis, price_history: np.ndarray):
        """發送 Discord 警報"""
        import discord
        
        if signal.signal_type in [SignalType.STRONG_BUY, SignalType.BUY]:
            color = discord.Color.green()
        elif signal.signal_type in [SignalType.STRONG_SELL, SignalType.SELL]:
            color = discord.Color.red()
        else:
            color = discord.Color.light_grey()
            
        embed = discord.Embed(
            title=f"{signal.signal_type.value} {signal.symbol}USDT",
            description=f"**Price:** ${signal.current_price:,.2f}\n**AI Score:** {ai_analysis.validity_score:.0f}/100",
            color=color,
            timestamp=datetime.now()
        )
        
        pred_path_str = " -> ".join([f"${p:.2f}" for p in signal.predicted_prices])
        embed.add_field(name="🔮 5-Step Prediction", value=f"`{pred_path_str}`", inline=False)
        
        embed.add_field(name="🎯 Entry", value=f"${signal.entry_price:,.2f}", inline=True)
        embed.add_field(name="💰 Take Profit", value=f"${signal.take_profit:,.2f}", inline=True)
        embed.add_field(name="🛑 Stop Loss", value=f"${signal.stop_loss:,.2f}", inline=True)
        
        embed.add_field(name="🤖 AI Reasoning", value=f"*{ai_analysis.reasoning}*", inline=False)
        embed.add_field(name="📊 Market", value=ai_analysis.market_condition, inline=True)
        embed.add_field(name="📉 Risk/Reward", value=f"{signal.risk_reward_ratio:.2f}", inline=True)
        
        embed.set_footer(text="Crypto Price Predictor • AI Enhanced")
        
        chart_buf = generate_prediction_chart(signal.symbol, price_history, signal.predicted_prices)
        file = None
        if chart_buf:
            file = discord.File(chart_buf, filename="prediction.png")
            embed.set_image(url="attachment://prediction.png")
        
        self.discord_bot.queue_embed(embed, file)
        logger.info(f"✅ Signal queued for Discord for {signal.symbol}")

    def _update_global_signal_state(self, signal: TradingSignal, ai_analysis: GeminiAnalysis, filtered: bool = False):
        """更新全局信號狀態供 !portfolio 使用"""
        final_pred_price = signal.predicted_prices[-1] if signal.predicted_prices else signal.current_price
        
        signal_data = {
            'symbol': signal.symbol,
            'signal_type': signal.signal_type.value if not filtered else "⚪ WAIT",
            'current_price': signal.current_price,
            'predicted_price': final_pred_price,
            'confidence': signal.confidence,
            'ai_validity': ai_analysis.validity_score,
            'trend_direction': signal.trend_direction.value,
            'rsi': signal.technical_indicators.get('rsi', 50),
            'entry_price': signal.entry_price,
            'take_profit': signal.take_profit,
            'stop_loss': signal.stop_loss,
            'timestamp': datetime.now().isoformat()
        }
        self.discord_bot.update_signal(signal.symbol, signal_data)

    def _create_portfolio_update(self):
        """創建投資組合狀態 Embed"""
        import discord
        embed = discord.Embed(
            title="📊 Market Overview Update",
            description=f"Generated {len(self.symbols)} signals",
            color=discord.Color.blue(),
            timestamp=datetime.now()
        )
        return embed


if __name__ == "__main__":
    # Check if should use HuggingFace
    use_hf = os.getenv('USE_HUGGINGFACE_MODELS', 'true').lower() == 'true'
    logger.info(f"🎯 Starting bot with HuggingFace: {use_hf}")
    
    bot = RealtimeTradingBot(use_huggingface=use_hf)
    bot.run()
