#!/usr/bin/env python3
"""Script to upload trained models to Hugging Face Hub."""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.huggingface_model_manager import HuggingFaceModelManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Upload trained models to HuggingFace."""
    
    # Load environment variables
    load_dotenv()
    
    # Configuration
    local_models_dir = "models/saved_models"
    hf_repo_id = os.getenv('HUGGINGFACE_REPO_ID', 'zongowo111/crypto_model')
    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    
    # Validation
    if not hf_token:
        logger.error("❌ HUGGINGFACE_TOKEN not found in environment variables")
        logger.error("Please set HUGGINGFACE_TOKEN in your .env file or system environment")
        sys.exit(1)
    
    if not Path(local_models_dir).exists():
        logger.error(f"❌ Models directory not found: {local_models_dir}")
        sys.exit(1)
    
    model_files = list(Path(local_models_dir).glob("*.pth"))
    if not model_files:
        logger.error(f"❌ No .pth files found in {local_models_dir}")
        sys.exit(1)
    
    logger.info("🚀 Crypto Price Predictor - Model Upload to HuggingFace")
    logger.info("="*70)
    logger.info(f"📑 Models to upload: {len(model_files)}")
    for f in model_files:
        logger.info(f"  - {f.name}")
    logger.info(f"📤 Target repository: {hf_repo_id}")
    logger.info("="*70)
    
    # Upload
    manager = HuggingFaceModelManager(repo_id=hf_repo_id)
    success = manager.upload_models(local_dir=local_models_dir)
    
    if success:
        logger.info("")
        logger.info("✅ Upload successful!")
        logger.info(f"🔗 Models available at: https://huggingface.co/{hf_repo_id}")
        logger.info("")
        logger.info("🔍 Next steps:")
        logger.info("  1. Set environment variable: HUGGINGFACE_REPO_ID=zongowo111/crypto_model")
        logger.info("  2. Run the bot with: USE_HUGGINGFACE_MODELS=true python run_bot.py")
        logger.info("")
        sys.exit(0)
    else:
        logger.error("❌ Upload failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
