# HuggingFace Model Setup Guide

This guide explains how to upload your trained models to HuggingFace Hub and configure the bot to download them automatically.

## Table of Contents
- [Why Use HuggingFace?](#why-use-huggingface)
- [Setup Steps](#setup-steps)
- [Upload Models](#upload-models)
- [Configure Bot](#configure-bot)
- [Troubleshooting](#troubleshooting)

---

## Why Use HuggingFace?

✅ **Benefits:**
- **Small GitHub**: Keep your GitHub repository small (no large model files)
- **Easy Distribution**: Share models with a simple URL
- **Version Control**: Track model versions automatically
- **Auto Download**: Bot downloads models on first run
- **Free Storage**: 90GB free per user
- **CDN**: Fast downloads from global servers

---

## Setup Steps

### Step 1: Create HuggingFace Account

1. Go to https://huggingface.co/
2. Click "Sign up"
3. Complete registration (email confirmation required)

### Step 2: Create API Token

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Select "Write" access (for uploads)
4. Click "Generate token"
5. Copy the token (starts with `hf_`)

### Step 3: Create Repository

1. Go to https://huggingface.co/new
2. Fill in:
   - **Repository name**: `crypto_model` (or your preference)
   - **Repository type**: Model
   - **Visibility**: Public (recommended for easy sharing)
   - Click "Create repository"

**Your repository ID will be**: `your_username/crypto_model`

### Step 4: Set Environment Variables

Create/update your `.env` file:

```bash
# HuggingFace Configuration
USE_HUGGINGFACE_MODELS=true
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx  # Your write token from Step 2
HUGGINGFACE_REPO_ID=your_username/crypto_model  # Your repo from Step 3

# Other settings...
```

---

## Upload Models

### Prerequisites

- Trained models in `models/saved_models/` directory
- HuggingFace token in `.env` file
- Internet connection

### Method 1: Using Upload Script (Recommended)

```bash
python upload_to_huggingface.py
```

**Output example:**
```
🏗️  Target Repository: zongowo111/crypto_model
🧠 Found 10 trained models:
  - BTC_lstm_model.pth (4.32 MB)
  - ETH_lstm_model.pth (4.32 MB)
  - BNB_lstm_model.pth (4.32 MB)
  ... (7 more models)

🚀 Starting upload to zongowo111/crypto_model...
✅ Uploaded BTC_lstm_model.pth
✅ Uploaded ETH_lstm_model.pth
...

✅ Upload successful!
🔗 View your repository: https://huggingface.co/zongowo111/crypto_model
```

### Method 2: Using Python Code

```python
from src.huggingface_model_manager import upload_trained_models

# Upload all models
success = upload_trained_models(
    local_dir="models/saved_models",
    repo_id="your_username/crypto_model"
)

if success:
    print("✅ Models uploaded successfully!")
else:
    print("❌ Upload failed")
```

### Method 3: Web Upload

1. Visit https://huggingface.co/your_username/crypto_model
2. Click "Upload file"
3. Select `.pth` files from `models/saved_models/`
4. Upload them one by one

---

## Configure Bot

### Option A: Auto Download (Recommended)

The bot automatically downloads models on startup:

```bash
# Make sure .env has:
USE_HUGGINGFACE_MODELS=true
HUGGINGFACE_REPO_ID=your_username/crypto_model

# Start the bot
python -m src.realtime_trading_bot
```

**First run output:**
```
🎯 Starting bot with HuggingFace: True
🔧 Model Source: HuggingFace Hub
📥 Downloading BTC model from HuggingFace...
✅ Downloaded BTC model to ...
... (downloading all 10 models)
🚀 Starting real-time trading bot monitoring...
```

### Option B: Use Local Models

If you prefer to keep models locally:

```bash
# In .env file:
USE_HUGGINGFACE_MODELS=false

# Bot will use models from:
# models/saved_models/BTC_lstm_model.pth
# models/saved_models/ETH_lstm_model.pth
# ... etc
```

### Option C: Manual Download

Download models programmatically:

```python
from src.huggingface_model_manager import download_crypto_models

# Download all models
models = download_crypto_models(
    repo_id="your_username/crypto_model"
)

print(models)  # {"BTC": "/path/to/BTC_lstm_model.pth", ...}
```

---

## Troubleshooting

### Error: "HUGGINGFACE_TOKEN not found"

**Solution:**
1. Check `.env` file has `HUGGINGFACE_TOKEN=hf_...`
2. Restart Python/bot to reload environment
3. Verify token at https://huggingface.co/settings/tokens

### Error: "Repository not found"

**Solution:**
1. Verify `HUGGINGFACE_REPO_ID` format: `username/repo-name`
2. Check repository exists at https://huggingface.co/your_username/crypto_model
3. Ensure token has write access

### Error: "Model file not found on hub"

**Solution:**
1. Upload models first: `python upload_to_huggingface.py`
2. Check models appear at: https://huggingface.co/your_username/crypto_model
3. Verify filenames match: `{SYMBOL}_lstm_model.pth`

### Slow Download Speed

**Solutions:**
1. Models cache locally after first download
2. Use hardwired internet (WiFi can be slower)
3. Try again during off-peak hours
4. Check internet connection: `ping huggingface.co`

### Upload Fails

**Checklist:**
- [ ] Token has "Write" access (not just "Read")
- [ ] Repository exists and is public
- [ ] Models exist in `models/saved_models/`
- [ ] Internet connection is stable
- [ ] File permissions allow reading models

**Manual upload:**
```bash
# Using git-lfs (if upload script fails)
git clone https://huggingface.co/your_username/crypto_model
cd crypto_model
cp ../models/saved_models/*.pth .
git lfs add *.pth
git add .
git commit -m "Add trained models"
git push
```

---

## Advanced Usage

### Load Single Model

```python
from src.huggingface_model_manager import HuggingFaceModelManager
import torch

manager = HuggingFaceModelManager(
    repo_id="your_username/crypto_model"
)

# Download and load BTC model
model = manager.load_model_from_hf(
    symbol="BTC",
    device=torch.device('cpu'),
    model_type="lstm"
)

if model:
    print("✅ Model loaded")
    # Use model for predictions
else:
    print("❌ Failed to load model")
```

### Get Repository Info

```python
from src.huggingface_model_manager import HuggingFaceModelManager

manager = HuggingFaceModelManager()
info = manager.get_model_info()

print(info)
# {
#   'repo_id': 'your_username/crypto_model',
#   'url': 'https://huggingface.co/your_username/crypto_model',
#   'private': False,
#   'last_modified': '2025-12-13T08:39:47Z',
#   'files_count': 10
# }
```

### Update Models

To update models after retraining:

```bash
# 1. Retrain models locally
python local_train.py

# 2. Upload new versions
python upload_to_huggingface.py

# 3. Bot will auto-download updates on next restart
```

---

## Security Considerations

⚠️ **Important:**

1. **Never commit tokens to Git**
   - Use `.env` file
   - Add `.env` to `.gitignore`

2. **Regenerate token if exposed**
   - Go to https://huggingface.co/settings/tokens
   - Delete and recreate token

3. **Repository Visibility**
   - Set to "Public" for easy sharing
   - Or "Private" if you want restricted access

---

## Support

- HuggingFace Docs: https://huggingface.co/docs
- HF Hub Python: https://huggingface.co/docs/huggingface_hub
- Issues/Questions: Create GitHub issue

---

**Congratulations!** Your models are now on HuggingFace Hub! 🎉
