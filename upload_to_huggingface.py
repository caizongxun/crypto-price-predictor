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
    """Upload trained models to HuggingFace using batch folder upload."""
    
    # Check HuggingFace token
    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    if not hf_token:
        logger.error("\u274c HUGGINGFACE_TOKEN not found in environment!")
        logger.info("📋 Set it with: export HUGGINGFACE_TOKEN=\"your_token_here\"")
        return False
    
    # Get repository ID
    repo_id = os.getenv('HUGGINGFACE_REPO_ID', 'zongowo111/crypto_model')
    logger.info(f"🏗️  Target Repository: {repo_id}")
    
    # Check local models directory
    models_dir = Path('models/saved_models')
    if not models_dir.exists():
        logger.error(f"\u274c Models directory not found: {models_dir}")
        return False
    
    model_files = list(models_dir.glob('*.pth'))
    if not model_files:
        logger.error(f"\u274c No .pth model files found in {models_dir}")
        return False
    
    total_size_mb = sum(f.stat().st_size for f in model_files) / (1024**2)
    logger.info(f"🧰 Found {len(model_files)} trained models:")
    for f in sorted(model_files):
        logger.info(f"  - {f.name} ({f.stat().st_size / (1024**2):.2f} MB)")
    logger.info(f"\ud83d� Total size: {total_size_mb:.2f} MB")
    
    # Import HuggingFace API
    try:
        from huggingface_hub import HfApi, login
    except ImportError as e:
        logger.error(f"\u274c Failed to import HuggingFace API: {e}")
        logger.info("📋 Install with: pip install huggingface_hub")
        return False
    
    # Login to HuggingFace
    try:
        login(token=hf_token)
        logger.info("✅ Logged in to HuggingFace")
    except Exception as e:
        logger.error(f"\u274c Failed to login to HuggingFace: {e}")
        return False
    
    # Initialize API
    api = HfApi(token=hf_token)
    
    # Create repo if doesn't exist
    try:
        logger.info(f"🔧 Creating/verifying repository...")
        api.create_repo(repo_id=repo_id, exist_ok=True, private=False, repo_type="model")
        logger.info(f"✅ Repository {repo_id} ready")
    except Exception as e:
        logger.warning(f"Repository creation info: {e}")
    
    # Upload entire folder at once (batch upload)
    logger.info(f"\n🚀 Starting batch upload of {len(model_files)} models to {repo_id}...")
    logger.info(f"This uses optimized batch upload to avoid API rate limits.\n")
    
    try:
        # Use upload_folder instead of uploading files one by one
        # This is much more efficient and avoids API rate limiting
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(models_dir),
            path_in_repo="models",  # Upload to 'models/' subfolder on HF
            repo_type="model",
            multi_commit=True,  # Use multiple commits for large uploads
            multi_commit_pr=False,  # Don't create PR, direct commit
        )
        
        logger.info(f"\n✅ Upload successful!")
        logger.info(f"🔗 View your repository: https://huggingface.co/{repo_id}")
        logger.info(f"🧰 {len(model_files)} models uploaded ({total_size_mb:.2f} MB total)")
        return True
        
    except Exception as e:
        logger.error(f"\n\u274c Upload failed: {e}", exc_info=True)
        logger.info("\n📝 Troubleshooting tips:")
        logger.info("  1. Check token has 'Write' permission")
        logger.info("  2. Verify repository exists and is public")
        logger.info("  3. Check internet connection to huggingface.co")
        logger.info("  4. Try again in a few minutes if rate-limited")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
