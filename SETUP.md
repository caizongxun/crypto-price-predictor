# Setup Guide - Cryptocurrency Price Predictor

## Prerequisites

- Python 3.9+
- Git
- Discord Bot Account
- (Optional) Binance API Keys for direct data access

## Installation Steps

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/crypto-price-predictor.git
cd crypto-price-predictor
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your credentials:
- `DISCORD_BOT_TOKEN`: Your Discord bot token
- `DISCORD_CHANNEL_ID`: Target Discord channel ID
- `BINANCE_API_KEY` (optional): Your Binance API key
- `BINANCE_API_SECRET` (optional): Your Binance API secret

### 5. Setup Configuration

Edit `config/config.yaml` and customize:
- Cryptocurrencies to monitor
- Model settings
- Trading parameters
- Discord settings

### 6. Train Model (Optional)

If you want to train the model with your own data:

```bash
python train_model.py --symbol BTC --model lstm --epochs 100
```

### 7. Run Application

```bash
python main.py
```

The application will start monitoring cryptocurrencies and send signals to Discord.

## Docker Deployment

### Build and Run with Docker

```bash
docker-compose up -d
```

### View Logs

```bash
docker-compose logs -f crypto-predictor
```

### Stop Application

```bash
docker-compose down
```

## Discord Bot Setup

### 1. Create Discord Application

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Click "New Application"
3. Give it a name (e.g., "Crypto Predictor")
4. Go to "Bot" section and click "Add Bot"
5. Copy the token and add to `.env`

### 2. Set Permissions

Under "OAuth2" → "URL Generator":
- Scopes: `bot`
- Permissions:
  - `Send Messages`
  - `Embed Links`
  - `Read Messages/View Channels`
  - `Mention @everyone, @here, and All Roles`

### 3. Get Channel ID

1. Enable Developer Mode in Discord
2. Right-click channel → Copy Channel ID
3. Add to `.env` as `DISCORD_CHANNEL_ID`

### 4. Invite Bot

Use the generated OAuth2 URL from step 2 to invite bot to your server.

## Model Training

### Train LSTM Model

```bash
python train_model.py \
  --symbol BTC \
  --model lstm \
  --epochs 100 \
  --lookback 60
```

### Train Transformer Model

```bash
python train_model.py \
  --symbol BTC \
  --model transformer \
  --epochs 100
```

## Troubleshooting

### Discord Bot Not Sending Messages

1. Check if bot token is correct
2. Verify channel ID is accurate
3. Ensure bot has message permissions
4. Check logs for error messages

### Data Fetching Issues

1. Verify internet connection
2. Check API rate limits
3. Ensure correct trading pairs (e.g., BTC/USDT)
4. Try with public data sources first

### Model Training Issues

1. Ensure sufficient historical data (500+ candles)
2. Check GPU availability (if using CUDA)
3. Adjust batch size if memory issues occur
4. Verify feature preparation

### Performance Issues

1. Reduce number of cryptocurrencies monitored
2. Increase update interval
3. Use GPU if available
4. Optimize model size

## Monitoring and Maintenance

### Check Logs

```bash
tail -f logs/crypto_predictor.log
```

### Monitor Resource Usage

```bash
# With Docker
docker stats crypto-predictor
```

### Update Models

Retrain models periodically with latest data:

```bash
python train_model.py --symbol BTC --epochs 50
```

## Advanced Configuration

See `config/config.yaml` for detailed configuration options including:
- Model hyperparameters
- Technical indicator settings
- Trading signal thresholds
- Data source preferences

## Support

For issues or questions:
1. Check the README.md
2. Review logs in `logs/` directory
3. Open an issue on GitHub
4. Check Discord bot permissions
