"""Main entry point for the cryptocurrency price predictor."""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Dict

import torch
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv

from src.data_fetcher import DataFetcher
from src.model_trainer import ModelTrainer
from src.predictor import Predictor
from src.technical_analysis import TechnicalAnalyzer
from src.discord_bot import setup_discord_bot, start_discord_bot, DiscordBot
from src.utils import (
    setup_logging, load_config, create_directories, 
    validate_config, get_project_root
)

load_dotenv()

# Setup logging
setup_logging(
    log_level=os.getenv('LOG_LEVEL', 'INFO'),
    log_file='logs/crypto_predictor.log'
)
logger = logging.getLogger(__name__)

# Create directories
create_directories()

# Load configuration
config = load_config('config/config.yaml')

# Validate configuration
if not validate_config(config):
    logger.error("Invalid configuration")
    exit(1)


class CryptoPricePredictor:
    """Main application class."""
    
    def __init__(self, config: Dict):
        """Initialize the application.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_fetcher = DataFetcher(config)
        self.predictor = None
        self.discord_bot = None
        self.scheduler = None
        self.cryptocurrencies = config.get('cryptocurrencies', [])
        
        logger.info("Cryptocurrency Price Predictor initialized")
    
    async def initialize_models(self):
        """Initialize prediction models."""
        try:
            model_type = self.config.get('model', {}).get('type', 'lstm')
            model_path = f"models/saved_models/best_{model_type}_model.pth"
            
            self.predictor = Predictor(
                model_path=model_path,
                model_type=model_type,
                lookback=self.config.get('model', {}).get('lookback_period', 60),
                prediction_horizon=self.config.get('model', {}).get('prediction_horizon', 7),
                config=self.config
            )
            logger.info(f"Predictor initialized with {model_type} model")
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
    
    async def initialize_discord_bot(self):
        """Initialize Discord bot."""
        try:
            token = os.getenv('DISCORD_BOT_TOKEN')
            if not token:
                logger.warning("Discord bot token not found. Bot functionality disabled.")
                return
            
            self.discord_bot = await setup_discord_bot(self.config)
            if self.discord_bot:
                logger.info("Discord bot initialized")
                # Start bot in background
                asyncio.create_task(start_discord_bot(self.discord_bot, token))
        except Exception as e:
            logger.error(f"Failed to initialize Discord bot: {e}")
    
    async def fetch_and_predict(self, cryptocurrency: Dict):
        """Fetch data and generate predictions for a cryptocurrency.
        
        Args:
            cryptocurrency: Cryptocurrency configuration
        """
        try:
            symbol = cryptocurrency.get('symbol')
            trading_pair = cryptocurrency.get('trading_pair')
            
            logger.info(f"Fetching data for {symbol}...")
            
            # Fetch data
            df = self.data_fetcher.fetch_ohlcv_binance(
                trading_pair,
                timeframe='1d',
                limit=100
            )
            
            if df is None or df.empty:
                logger.warning(f"No data fetched for {symbol}")
                return
            
            # Add technical indicators
            df = self.data_fetcher.add_technical_indicators(df)
            
            # Generate trading signal
            if self.predictor:
                signal = self.predictor.generate_trading_signal(symbol, df)
                
                if signal and self.discord_bot:
                    # Send signal to Discord
                    cog = self.discord_bot.get_cog('DiscordBot')
                    if cog:
                        await cog.send_signal_notification(signal)
                        logger.info(f"Signal sent for {symbol}: {signal.get('recommendation')}")
        except Exception as e:
            logger.error(f"Error fetching and predicting for {symbol}: {e}")
            if self.discord_bot:
                cog = self.discord_bot.get_cog('DiscordBot')
                if cog:
                    await cog.send_error_notification(f"Error predicting {symbol}: {str(e)}")
    
    async def monitoring_loop(self):
        """Main monitoring loop."""
        try:
            logger.info("Starting monitoring loop")
            
            for cryptocurrency in self.cryptocurrencies:
                await self.fetch_and_predict(cryptocurrency)
                
                # Rate limiting
                await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
    
    async def schedule_predictions(self):
        """Schedule periodic predictions."""
        try:
            self.scheduler = AsyncIOScheduler()
            
            update_interval = self.config.get('discord', {}).get('update_interval', 3600)
            update_minutes = update_interval // 60
            
            # Schedule monitoring task
            self.scheduler.add_job(
                self.monitoring_loop,
                'interval',
                minutes=update_minutes,
                id='monitoring_job'
            )
            
            self.scheduler.start()
            logger.info(f"Scheduled predictions every {update_minutes} minutes")
        except Exception as e:
            logger.error(f"Failed to schedule predictions: {e}")
    
    async def run(self):
        """Run the application."""
        try:
            logger.info("Starting Cryptocurrency Price Predictor...")
            logger.info(f"Monitoring {len(self.cryptocurrencies)} cryptocurrencies")
            logger.info(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
            
            # Initialize components
            await self.initialize_models()
            await self.initialize_discord_bot()
            
            # Initial prediction run
            await self.monitoring_loop()
            
            # Schedule periodic predictions
            await self.schedule_predictions()
            
            # Keep running
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                logger.info("Shutdown signal received")
        except Exception as e:
            logger.error(f"Fatal error in main loop: {e}")
            raise


async def main():
    """Main entry point."""
    app = CryptoPricePredictor(config)
    await app.run()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        exit(1)
