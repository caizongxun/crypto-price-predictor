"""Crypto Price Predictor Package"""
__version__ = "1.0.0"
__author__ = "Crypto Predictor Team"
__email__ = "support@example.com"

from .data_fetcher import DataFetcher
from .model_trainer import ModelTrainer
from .predictor import Predictor
from .technical_analysis import TechnicalAnalyzer
from .discord_bot import DiscordBot

__all__ = [
    'DataFetcher',
    'ModelTrainer',
    'Predictor',
    'TechnicalAnalyzer',
    'DiscordBot'
]
