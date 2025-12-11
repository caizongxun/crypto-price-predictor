"""Script to train the prediction model."""

import os
import argparse
import logging
from datetime import datetime, timedelta

import torch
from dotenv import load_dotenv

from src.data_fetcher import DataFetcher
from src.model_trainer import ModelTrainer
from src.utils import setup_logging, create_directories

load_dotenv()
setup_logging(log_level='INFO', log_file='logs/training.log')
logger = logging.getLogger(__name__)


def train_model(symbol: str, trading_pair: str, model_type: str = 'lstm',
               lookback: int = 60, epochs: int = 100, batch_size: int = 32):
    """Train a prediction model for a cryptocurrency.
    
    Args:
        symbol: Cryptocurrency symbol (e.g., BTC)
        trading_pair: Trading pair (e.g., BTC/USDT)
        model_type: Type of model (lstm or transformer)
        lookback: Lookback period in days
        epochs: Number of training epochs
        batch_size: Batch size
    """
    try:
        create_directories()
        
        logger.info(f"Starting training for {symbol}")
        logger.info(f"Model: {model_type}, Lookback: {lookback} days, Epochs: {epochs}")
        
        # Fetch data
        logger.info("Fetching historical data...")
        data_fetcher = DataFetcher()
        df = data_fetcher.fetch_ohlcv_binance(trading_pair, timeframe='1d', limit=500)
        
        if df is None or df.empty:
            logger.error(f"Failed to fetch data for {symbol}")
            return False
        
        # Add technical indicators
        logger.info("Adding technical indicators...")
        df = data_fetcher.add_technical_indicators(df)
        
        # Prepare features
        logger.info("Preparing ML features...")
        X, y, scaler = data_fetcher.prepare_ml_features(df, lookback=lookback)
        
        if X is None:
            logger.error("Failed to prepare features")
            return False
        
        # Create trainer
        config = {
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'd_model': 512,
            'num_heads': 8
        }
        
        trainer = ModelTrainer(model_type=model_type, config=config)
        
        # Prepare data
        X_train, X_test, y_train, y_test = trainer.prepare_data(
            X, y, test_size=0.2
        )
        
        if X_train is None:
            logger.error("Failed to prepare training data")
            return False
        
        # Create model
        trainer.create_model(input_size=X_train.shape[2])
        
        # Train model
        logger.info(f"Training {model_type.upper()} model...")
        history = trainer.train(
            X_train, y_train, X_test, y_test,
            epochs=epochs,
            batch_size=batch_size,
            early_stopping_patience=15
        )
        
        if history is None:
            logger.error("Training failed")
            return False
        
        # Save model
        model_path = f"models/saved_models/{symbol}_{model_type}_model.pth"
        trainer.save_model(model_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info("Training completed successfully")
        return True
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train cryptocurrency prediction model')
    parser.add_argument('--symbol', default='BTC', help='Cryptocurrency symbol')
    parser.add_argument('--trading-pair', default='BTC/USDT', help='Trading pair')
    parser.add_argument('--model', default='lstm', choices=['lstm', 'transformer'], help='Model type')
    parser.add_argument('--lookback', type=int, default=60, help='Lookback period in days')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    success = train_model(
        symbol=args.symbol,
        trading_pair=args.trading_pair,
        model_type=args.model,
        lookback=args.lookback,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    exit(0 if success else 1)
