#!/usr/bin/env python
"""Upload trained models to HuggingFace Hub.

Usage:
    python upload_to_huggingface.py

Prerequisites:
    1. Set HUGGINGFACE_TOKEN environment variable with your HF write token
    2. Have trained model files in models/saved_models/
    3. Create repository at https://huggingface.co/new

Example:
    export HUGGINGFACE_TOKEN="hf_xxxxxx"
    python upload_to_huggingface.py
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def main():
    """Upload trained models to HuggingFace."""
    
    # Check HuggingFace token
    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    if not hf_token:
        logger.error("\u274c HUGGINGFACE_TOKEN not found in environment!")
        logger.info("📋 Set it with: export HUGGINGFACE_TOKEN=\"your_token_here\"")
        return False
    
    # Get repository ID
    repo_id = os.getenv('HUGGINGFACE_REPO_ID', 'zongowo111/crypto_model')
    logger.info(f"🏗️ Target Repository: {repo_id}")
    
    # Check local models directory
    models_dir = Path('models/saved_models')
    if not models_dir.exists():
        logger.error(f"\u274c Models directory not found: {models_dir}")
        return False
    
    model_files = list(models_dir.glob('*.pth'))
    if not model_files:
        logger.error(f"\u274c No .pth model files found in {models_dir}")
        return False
    
    logger.info(f"🧰 Found {len(model_files)} trained models:")
    for f in sorted(model_files):
        logger.info(f"  - {f.name} ({f.stat().st_size / (1024**2):.2f} MB)")
    
    # Import HuggingFace manager
    try:
        from src.huggingface_model_manager import HuggingFaceModelManager
    except ImportError as e:
        logger.error(f"\u274c Failed to import HuggingFaceModelManager: {e}")
        logger.info("📋 Install with: pip install huggingface_hub")
        return False
    
    # Create manager and upload
    logger.info(f"\n🚀 Starting upload to {repo_id}...\n")
    
    try:
        manager = HuggingFaceModelManager(repo_id=repo_id)
        success = manager.upload_models(str(models_dir))
        
        if success:
            logger.info(f"\n✅ Upload successful!")
            logger.info(f"🔗 View your repository: https://huggingface.co/{repo_id}")
            return True
        else:
            logger.error("\n\u274c Upload failed!")
            return False
            
    except Exception as e:
        logger.error(f"\n\u274c Upload error: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
