"""Crypto Price Predictor Package"""
__version__ = "1.0.0"
__author__ = "Crypto Predictor Team"
__email__ = "support@example.com"

from .data_fetcher_tft_v3 import TFTDataFetcher
from .model_tft_v3 import TemporalFusionTransformer
from .discord_bot import DiscordBot
from .discord_bot_handler import DiscordBotHandler
from .huggingface_model_manager import HuggingFaceModelManager
from .utils import *
from .plotting import *

__all__ = [
    'TFTDataFetcher',
    'TemporalFusionTransformer',
    'DiscordBot',
    'DiscordBotHandler',
    'HuggingFaceModelManager',
]
