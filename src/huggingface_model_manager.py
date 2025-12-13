"""Hugging Face Model Manager for cloud-based model storage and retrieval."""

import os
import logging
from typing import Optional, Dict
from pathlib import Path
import torch

try:
    from huggingface_hub import hf_hub_download, HfApi, login
except ImportError:
    raise ImportError(
        "huggingface_hub not installed. Install with: pip install huggingface_hub"
    )

logger = logging.getLogger(__name__)


class HuggingFaceModelManager:
    """Manages model download and upload from/to Hugging Face Hub."""
    
    def __init__(self, repo_id: str = "zongowo111/crypto_model", 
                 local_cache_dir: str = "models/saved_models"):
        """
        Initialize HuggingFace model manager.
        
        Args:
            repo_id: HuggingFace repository ID (e.g., "username/repo-name")
            local_cache_dir: Local directory to cache downloaded models
        """
        self.repo_id = repo_id
        self.local_cache_dir = Path(local_cache_dir)
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if HF token is available
        self.hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if self.hf_token:
            try:
                login(token=self.hf_token)
                logger.info("✅ Logged in to Hugging Face")
            except Exception as e:
                logger.warning(f"Failed to login to Hugging Face: {e}")
        else:
            logger.warning("⚠️ HUGGINGFACE_TOKEN not found. Public models only.")
    
    def download_model(self, symbol: str, model_type: str = "lstm") -> Optional[str]:
        """
        Download model from Hugging Face Hub.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., "BTC", "ETH")
            model_type: Type of model (e.g., "lstm")
            
        Returns:
            Path to downloaded model file, or None if download failed
        """
        try:
            model_filename = f"{symbol}_{model_type}_model.pth"
            local_model_path = self.local_cache_dir / model_filename
            
            # If already cached locally, return path
            if local_model_path.exists():
                logger.info(f"✅ Using cached model: {local_model_path}")
                return str(local_model_path)
            
            # Download from HF Hub
            logger.info(f"📥 Downloading {symbol} model from Hugging Face...")
            downloaded_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=model_filename,
                cache_dir=str(self.local_cache_dir),
                force_download=False,  # Use cache if available
            )
            
            logger.info(f"✅ Downloaded {symbol} model to {downloaded_path}")
            return downloaded_path
            
        except Exception as e:
            logger.error(f"❌ Failed to download {symbol} model: {e}")
            return None
    
    def upload_models(self, local_dir: str = "models/saved_models", 
                      repo_description: str = None) -> bool:
        """
        Upload all trained models to Hugging Face Hub.
        
        Args:
            local_dir: Local directory containing trained models
            repo_description: Description for the repository
            
        Returns:
            True if upload successful, False otherwise
        """
        if not self.hf_token:
            logger.error("❌ HUGGINGFACE_TOKEN not set. Cannot upload.")
            return False
        
        try:
            api = HfApi(token=self.hf_token)
            local_path = Path(local_dir)
            
            if not local_path.exists():
                logger.error(f"❌ Local directory not found: {local_dir}")
                return False
            
            # Get all .pth files
            model_files = list(local_path.glob("*.pth"))
            if not model_files:
                logger.error(f"❌ No .pth files found in {local_dir}")
                return False
            
            logger.info(f"📤 Uploading {len(model_files)} models to {self.repo_id}...")
            
            # Create repo if doesn't exist
            try:
                api.create_repo(repo_id=self.repo_id, exist_ok=True, private=False)
                logger.info(f"✅ Repository {self.repo_id} ready")
            except Exception as e:
                logger.warning(f"Repository creation info: {e}")
            
            # Upload each model file
            for model_file in model_files:
                try:
                    api.upload_file(
                        path_or_fileobj=str(model_file),
                        path_in_repo=model_file.name,
                        repo_id=self.repo_id,
                        repo_type="model"
                    )
                    logger.info(f"✅ Uploaded {model_file.name}")
                except Exception as e:
                    logger.error(f"❌ Failed to upload {model_file.name}: {e}")
            
            logger.info(f"✅ Successfully uploaded all models to {self.repo_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Upload failed: {e}")
            return False
    
    def load_model_from_hf(self, symbol: str, device: torch.device = None,
                          model_type: str = "lstm") -> Optional[torch.nn.Module]:
        """
        Download and load model from Hugging Face.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., "BTC")
            device: PyTorch device (cpu or cuda), auto-detect if None
            model_type: Type of model ("lstm" or "transformer")
            
        Returns:
            Loaded PyTorch model or None if loading failed
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        try:
            # Download model
            model_path = self.download_model(symbol, model_type)
            if not model_path:
                logger.error(f"❌ Failed to download {symbol} model from HF")
                return None
            
            # Import model classes
            from .model_trainer import LSTMModel, TransformerModel
            
            # Create model architecture - match your training config
            if model_type == "lstm":
                model = LSTMModel(
                    input_size=17,
                    hidden_size=256,
                    num_layers=3,
                    dropout=0.3,
                    output_size=5
                )
            elif model_type == "transformer":
                model = TransformerModel(
                    input_size=17,
                    d_model=256,
                    num_heads=8,
                    num_layers=3,
                    output_size=5
                )
            else:
                logger.error(f"❌ Unknown model type: {model_type}")
                return None
            
            # Load weights
            try:
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict, strict=False)
                model.to(device)
                model.eval()
                logger.info(f"✅ Loaded {symbol} {model_type} model from HF")
                return model
            except Exception as e:
                logger.error(f"❌ Failed to load state dict: {e}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to load model from HF: {e}")
            return None
    
    def get_model_info(self) -> Dict:
        """
        Get information about models in the repository.
        
        Returns:
            Dictionary with repository information
        """
        try:
            api = HfApi()
            repo_info = api.repo_info(repo_id=self.repo_id, repo_type="model")
            return {
                'repo_id': self.repo_id,
                'url': f"https://huggingface.co/{self.repo_id}",
                'private': repo_info.private,
                'last_modified': str(repo_info.last_modified),
                'files_count': len(repo_info.siblings) if repo_info.siblings else 0
            }
        except Exception as e:
            logger.error(f"❌ Failed to get repo info: {e}")
            return {}


# Convenience functions
def download_crypto_models(symbols: list = None, 
                          repo_id: str = "zongowo111/crypto_model") -> Dict[str, str]:
    """
    Download all cryptocurrency models from HuggingFace.
    
    Args:
        symbols: List of symbols to download (e.g., ["BTC", "ETH"])
        repo_id: HuggingFace repository ID
        
    Returns:
        Dictionary mapping symbols to local model paths
    """
    if symbols is None:
        symbols = ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOGE', 'MATIC', 'AVAX', 'LINK']
    
    manager = HuggingFaceModelManager(repo_id=repo_id)
    model_paths = {}
    
    for symbol in symbols:
        path = manager.download_model(symbol)
        if path:
            model_paths[symbol] = path
        else:
            logger.warning(f"⚠️ Could not download {symbol} model")
    
    return model_paths


def upload_trained_models(local_dir: str = "models/saved_models",
                         repo_id: str = "zongowo111/crypto_model") -> bool:
    """
    Upload trained models to HuggingFace.
    
    Args:
        local_dir: Local directory with trained models
        repo_id: Target HuggingFace repository
        
    Returns:
        True if successful, False otherwise
    """
    manager = HuggingFaceModelManager(repo_id=repo_id)
    return manager.upload_models(local_dir)
