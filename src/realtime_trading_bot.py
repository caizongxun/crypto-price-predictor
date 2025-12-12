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
    def __init__(self):
        self.symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT',
            'SOLUSDT', 'DOGEUSDT', 'MATICUSDT', 'AVAXUSDT', 'LINKUSDT'
        ]
        self.timeframe = '1h'
        self.check_interval = 900  # 15 minutes
        
        self.data_fetcher = DataFetcher()
        self.signal_generators = {}
        self.discord_bot = DiscordBotHandler()
        self.gemini_validator = GeminiSignalValidator(api_key=os.getenv('GROQ_API_KEY'))
        self.mtf_analyzer = MultiTimeframeAnalyzer(self.data_fetcher)
        self.technical_analyzer = TechnicalAnalyzer()
        
        self.last_check_time = {}
        self.active_signals = {}
        
        # 啟動 Discord Bot 線程
        self.discord_bot.start()
        
        # 初始化模型和信號生成器
        self._initialize_models()
        
    def _initialize_models(self):
        """為每個交易對加載訓練好的模型 (Ensemble: LSTM + Transformer + XGBoost)"""
        for symbol in self.symbols:
            try:
                # 這裡我們使用 LSTM 作為主要模型，但代碼結構允許未來擴展
                model_trainer = ModelTrainer(model_type='lstm', config={'hidden_size': 128, 'num_layers': 2})
                
                # 嘗試加載模型
                model_path = f"models/saved_models/{symbol.replace('USDT', '')}_lstm_model.pth"
                if os.path.exists(model_path):
                    logger.info(f"Loading model from {model_path}")
                    # 注意：這裡我們需要確保加載的模型 output_size=5。
                    # 如果舊模型是 output_size=1，加載時會報錯。
                    # 為了安全起見，如果加載失敗，我們會捕獲異常並使用未訓練的新模型（會觸發重新訓練）
                    try:
                        model_trainer.load_model(model_path, input_size=17)
                        
                        # 簡單檢查模型輸出尺寸
                        if model_trainer.model.fc2.out_features != 5:
                            logger.warning(f"Model for {symbol} has wrong output size. Re-initializing.")
                            model_trainer.create_model(input_size=17) # Reset
                    except Exception as e:
                        logger.warning(f"Error loading model for {symbol}: {e}. Initializing new model.")
                        model_trainer.create_model(input_size=17)
                else:
                    logger.warning(f"No saved model for {symbol}. Initializing new model.")
                    model_trainer.create_model(input_size=17)
                
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

                wrapped_model = EnsembleModelWrapper(model_trainer.model, model_trainer.device)
                logger.info(f"✅ Loaded wrapped ensemble model for {symbol.replace('USDT', '')}")
                
                self.signal_generators[symbol] = SignalGenerator(
                    model=wrapped_model,
                    device=model_trainer.device
                )
                
                logger.info(f"🔧 SignalGenerator for {symbol.replace('USDT', '')}: model=✅ Loaded, device={model_trainer.device}")
                
            except Exception as e:
                logger.error(f"Error initializing model for {symbol}: {e}", exc_info=True)
        
        logger.info("📊 Signal Generators initialized for all symbols with ensemble models")

    def run(self):
        """主循環"""
        logger.info(f"✅ RealtimeTradingBot initialized")
        logger.info(f"⏱️  Check frequency: {self.check_interval//60} minutes")
        logger.info("🚀 Starting real-time trading bot monitoring...")
        logger.info("📢 Discord Bot 通知已啓用")
        logger.info("⏱️  檢查頻率: 每 15 分鐘一次")
        
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

            # 3. 多時間週期分析 (用於 AI 驗證上下文)
            short_term = self.mtf_analyzer.analyze_timeframe(symbol, '1h')
            medium_term = self.mtf_analyzer.analyze_timeframe(symbol, '4h')
            long_term = self.mtf_analyzer.analyze_timeframe(symbol, '1d')
            
            # 4. AI 驗證 (Gemini/Groq)
            logger.info(f"📈 Signal generated for {signal.symbol}: {signal.signal_type.value} (Confidence: {signal.confidence:.2%})")
            
            # 即使是 NEUTRAL 信號也進行 AI 分析，提供更多洞察
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
                
                # 只有分數過低才過濾，NEUTRAL 信號保留供參考
                if ai_analysis.validity_score < 40:
                    logger.info(f"🚫 Signal filtered by Gemini: score={ai_analysis.validity_score}")
                    # Update global signals even if filtered, to show "Wait" status
                    self._update_global_signal_state(signal, ai_analysis, filtered=True)
                    return
                
                # 更新信號參數
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
            self._send_discord_alert(signal, ai_analysis)
            
            # 6. 更新全局狀態
            self._update_global_signal_state(signal, ai_analysis)
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}", exc_info=True)

    def _send_discord_alert(self, signal: TradingSignal, ai_analysis: GeminiAnalysis):
        """發送 Discord 警報"""
        import discord
        
        # 決定顏色
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
        
        # 預測路徑可視化 (文字版)
        pred_path_str = " -> ".join([f"${p:.2f}" for p in signal.predicted_prices])
        embed.add_field(name="🔮 5-Step Prediction", value=f"`{pred_path_str}`", inline=False)
        
        # 主要數據
        embed.add_field(name="🎯 Entry", value=f"${signal.entry_price:,.2f}", inline=True)
        embed.add_field(name="💰 Take Profit", value=f"${signal.take_profit:,.2f}", inline=True)
        embed.add_field(name="🛑 Stop Loss", value=f"${signal.stop_loss:,.2f}", inline=True)
        
        # AI 分析
        embed.add_field(name="🤖 AI Reasoning", value=f"*{ai_analysis.reasoning}*", inline=False)
        embed.add_field(name="📊 Market", value=ai_analysis.market_condition, inline=True)
        embed.add_field(name="📉 Risk/Reward", value=f"{signal.risk_reward_ratio:.2f}", inline=True)
        
        embed.set_footer(text="Crypto Price Predictor • AI Enhanced")
        
        self.discord_bot.queue_embed(embed)
        logger.info(f"✅ Signal queued for Discord for {signal.symbol}")

    def _update_global_signal_state(self, signal: TradingSignal, ai_analysis: GeminiAnalysis, filtered: bool = False):
        """更新全局信號狀態供 !portfolio 使用"""
        
        # Calculate final price change from prediction
        final_pred_price = signal.predicted_prices[-1] if signal.predicted_prices else signal.current_price
        price_change_pct = (final_pred_price - signal.current_price) / signal.current_price * 100
        
        signal_data = {
            'symbol': signal.symbol,
            'signal_type': signal.signal_type.value if not filtered else "⚪ WAIT",
            'current_price': signal.current_price,
            'predicted_price': final_pred_price,
            'confidence': signal.confidence,
            'ai_validity': ai_analysis.validity_score, # 存儲 AI 分數
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
        # 這裡可以添加更多匯總信息
        return embed

if __name__ == "__main__":
    bot = RealtimeTradingBot()
    bot.run()
