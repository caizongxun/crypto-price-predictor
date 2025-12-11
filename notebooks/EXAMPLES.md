# Usage Examples

## 1. Quick Start

### Setup

```python
from src.data_fetcher import DataFetcher
from src.predictor import Predictor
from src.technical_analysis import TechnicalAnalyzer

# Initialize components
data_fetcher = DataFetcher()
predictor = Predictor('models/saved_models/best_lstm_model.pth')
analyzer = TechnicalAnalyzer()
```

### Fetch Real-time Data

```python
# Fetch data for Bitcoin
df = data_fetcher.fetch_ohlcv_binance('BTC/USDT', timeframe='1d', limit=100)

# Add technical indicators
df = data_fetcher.add_technical_indicators(df)

print(df.tail())
```

### Generate Price Prediction

```python
# Single step prediction
prediction = predictor.predict_price(df)

print(f"Current Price: ${prediction['current_price']:.2f}")
print(f"Predicted Price: ${prediction['predicted_price']:.2f}")
print(f"Change: {prediction['price_change_percent']:+.2f}%")
print(f"Confidence: {prediction['confidence']:.1f}%")
```

### Multi-step Prediction (7-day path)

```python
# Generate 7-day price path
price_path = predictor.predict_path(df, steps=7)

for i, pred in enumerate(price_path, 1):
    print(f"Day {i}: ${pred['predicted_price']:.2f}")
```

### Generate Trading Signal

```python
# Complete trading signal with support/resistance
signal = predictor.generate_trading_signal('BTC', df)

print(f"Signal: {signal['recommendation']}")
print(f"Entry Zone: ${signal['entry_min']:.2f} - ${signal['entry_max']:.2f}")
print(f"Support: ${signal['support_level']:.2f}")
print(f"Resistance: ${signal['resistance_level']:.2f}")
print(f"Stop Loss: ${signal['stop_loss']:.2f}")
print(f"Take Profits: {signal['take_profit']}")
```

## 2. Technical Analysis

### Find Support and Resistance

```python
sr_levels = analyzer.find_support_resistance(df, period=20)

print(f"Support: ${sr_levels['support']:.2f}")
print(f"Resistance: ${sr_levels['resistance']:.2f}")
print(f"Support Levels: {sr_levels['support_levels']}")
print(f"Resistance Levels: {sr_levels['resistance_levels']}")
```

### Detect RSI Divergence

```python
divergence = analyzer.detect_divergence(df)

if divergence.get('bullish_divergence'):
    print("Bullish Divergence Detected!")
elif divergence.get('bearish_divergence'):
    print("Bearish Divergence Detected!")
```

### Detect Breakouts

```python
breakout = analyzer.detect_breakout(df, period=20)

if breakout.get('breakout_up'):
    print(f"Breakout Up! Price above {breakout['high_level']:.2f}")
elif breakout.get('breakout_down'):
    print(f"Breakout Down! Price below {breakout['low_level']:.2f}")
```

### Get All Technical Signals

```python
signals = analyzer.get_signals(df)

print(f"RSI: {signals['rsi']:.1f}")
print(f"MACD: {signals['macd']:+.4f}")
print(f"Trend: {signals['trend']}")
print(f"Volatility: {signals['volatility']:.2f}%")
print(f"Overall Signal: {signals['overall_signal']}")
```

## 3. Monitor Multiple Cryptocurrencies

```python
cryptos = ['BTC', 'ETH', 'SOL', 'XRP', 'ADA']
trading_pairs = {
    'BTC': 'BTC/USDT',
    'ETH': 'ETH/USDT',
    'SOL': 'SOL/USDT',
    'XRP': 'XRP/USDT',
    'ADA': 'ADA/USDT'
}

signals = {}

for symbol in cryptos:
    try:
        # Fetch data
        df = data_fetcher.fetch_ohlcv_binance(
            trading_pairs[symbol],
            timeframe='1d',
            limit=100
        )
        
        # Add indicators
        df = data_fetcher.add_technical_indicators(df)
        
        # Generate signal
        signal = predictor.generate_trading_signal(symbol, df)
        signals[symbol] = signal
        
    except Exception as e:
        print(f"Error for {symbol}: {e}")

# Display results
for symbol, signal in signals.items():
    print(f"\n{symbol}:")
    print(f"  Recommendation: {signal['recommendation']}")
    print(f"  Confidence: {signal['confidence']:.1f}%")
    print(f"  Current: ${signal['current_price']:.2f}")
    print(f"  Predicted: ${signal['predicted_price']:.2f}")
```

## 4. Discord Integration

```python
import asyncio
from src.discord_bot import setup_discord_bot, DiscordBot

async def send_signals_to_discord():
    # Setup bot
    bot = await setup_discord_bot()
    
    # Get bot cog
    cog = bot.get_cog('DiscordBot')
    
    # Generate signals
    signals = {}  # ... generate signals as above
    
    # Send to Discord
    for symbol, signal in signals.items():
        if signal and signal['recommendation'] != 'HOLD':
            await cog.send_signal_notification(signal)
            await asyncio.sleep(1)  # Rate limiting

# Run async function
asyncio.run(send_signals_to_discord())
```

## 5. Model Training

```python
from src.model_trainer import ModelTrainer
from src.data_fetcher import DataFetcher

# Fetch and prepare data
data_fetcher = DataFetcher()
df = data_fetcher.fetch_ohlcv_binance('BTC/USDT', timeframe='1d', limit=500)
df = data_fetcher.add_technical_indicators(df)
X, y, scaler = data_fetcher.prepare_ml_features(df, lookback=60)

# Create trainer
config = {
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.2,
    'learning_rate': 0.001
}

trainer = ModelTrainer(model_type='lstm', config=config)

# Prepare data
X_train, X_test, y_train, y_test = trainer.prepare_data(X, y)

# Create and train model
trainer.create_model(input_size=X_train.shape[2])
history = trainer.train(
    X_train, y_train, X_test, y_test,
    epochs=100,
    batch_size=32
)

# Save model
trainer.save_model('models/saved_models/custom_lstm_model.pth')
```

## 6. Real-time Market Data

```python
# Get current price
realtimeBTC = data_fetcher.get_real_time_price('BTC/USDT')
print(f"BTC: ${realtime_btc:.2f}")

# Get comprehensive market data
market_data = data_fetcher.get_market_data('BTC/USDT')
print(f"Last: ${market_data['last']:.2f}")
print(f"Bid: ${market_data['bid']:.2f}")
print(f"Ask: ${market_data['ask']:.2f}")
print(f"Volume: {market_data['volume']:.2f}")
```

## 7. Portfolio Monitoring

```python
import pandas as pd

# Track multiple cryptocurrencies
portfolio = {
    'BTC': 0.5,    # holdings in BTC
    'ETH': 5.0,    # holdings in ETH
    'SOL': 100.0   # holdings in SOL
}

# Get current prices
prices = {}
for symbol in portfolio.keys():
    pair = f"{symbol}/USDT"
    prices[symbol] = data_fetcher.get_real_time_price(pair)

# Calculate portfolio value
total_value = sum(portfolio[symbol] * prices[symbol] for symbol in portfolio)

# Display portfolio
for symbol, holding in portfolio.items():
    value = holding * prices[symbol]
    percentage = (value / total_value) * 100
    print(f"{symbol}: {holding:.2f} @ ${prices[symbol]:.2f} = ${value:.2f} ({percentage:.1f}%)")

print(f"\nTotal Portfolio Value: ${total_value:.2f}")
```

## 8. Backtesting (Future Feature)

```python
# Example of how backtesting will work
from src.backtester import Backtester

backtester = Backtester(predictor, data_fetcher)
results = backtest.run(
    symbol='BTC',
    start_date='2023-01-01',
    end_date='2024-01-01',
    strategy='ensemble'
)

print(f"Total Return: {results['total_return']:.2f}%")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
