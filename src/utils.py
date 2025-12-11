"""Utility functions for the crypto predictor application."""

import logging
import os
from pathlib import Path
from typing import Dict, Any

import yaml

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = 'INFO', log_file: str = None):
    """Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level))
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")
    
    logger.info(f"Logging setup complete. Level: {log_level}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        if not os.path.exists(config_path):
            logger.warning(f"Config file not found: {config_path}")
            return {}
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Config loaded from {config_path}")
        return config or {}
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}


def save_config(config: Dict, config_path: str):
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info(f"Config saved to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save config: {e}")


def create_directories():
    """Create necessary project directories."""
    directories = [
        'logs',
        'models/saved_models',
        'data/historical',
        'config',
        'results'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    logger.info(f"Created {len(directories)} directories")


def format_price(price: float, decimals: int = 2) -> str:
    """Format price for display.
    
    Args:
        price: Price value
        decimals: Number of decimal places
        
    Returns:
        Formatted price string
    """
    return f"${price:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format percentage for display.
    
    Args:
        value: Percentage value
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    symbol = '+' if value > 0 else ''
    return f"{symbol}{value:.{decimals}f}%"


def get_project_root() -> Path:
    """Get project root directory.
    
    Returns:
        Path to project root
    """
    return Path(__file__).parent.parent


def validate_config(config: Dict) -> bool:
    """Validate configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_keys = ['discord', 'cryptocurrencies', 'model']
    
    for key in required_keys:
        if key not in config:
            logger.warning(f"Missing required config key: {key}")
            return False
    
    return True
