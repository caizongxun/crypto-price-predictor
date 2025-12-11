# API Reference

## DataFetcher

### fetch_ohlcv_binance(symbol, timeframe, limit)

Fetch OHLCV data from Binance.

**Parameters:**
- `symbol` (str): Trading pair (e.g., 'BTC/USDT')
- `timeframe` (str): Candlestick timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
- `limit` (int): Number of candles to fetch (default: 500)

**Returns:** pandas.DataFrame with OHLCV data

**Example:**
```python
df = data_fetcher.fetch_ohlcv_binance('BTC/USDT', '1d', limit=100)
```

### add_technical_indicators(df)

Add technical indicators to OHLCV data.

**Parameters:**
- `df` (pandas.DataFrame): DataFrame with OHLCV data

**Returns:** pandas.DataFrame with added indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, etc.)

**Example:**
```python
df = data_fetcher.add_technical_indicators(df)
```

### prepare_ml_features(df, lookback)

Prepare features for machine learning models.

**Parameters:**
- `df` (pandas.DataFrame): DataFrame with OHLCV and technical indicators
- `lookback` (int): Lookback period in days (default: 60)

**Returns:** Tuple of (X features, y labels, scaler)

**Example:**
```python
X, y, scaler = data_fetcher.prepare_ml_features(df, lookback=60)
```

## Predictor

### predict_price(df)

Predict cryptocurrency price.

**Parameters:**
- `df` (pandas.DataFrame): DataFrame with OHLCV and technical indicators

**Returns:** Dict with prediction results

**Example:**
```python
prediction = predictor.predict_price(df)
print(prediction['predicted_price'])
```

### predict_path(df, steps)

Generate multi-step price prediction.

**Parameters:**
- `df` (pandas.DataFrame): DataFrame with OHLCV data
- `steps` (int): Number of steps to predict (default: 7)

**Returns:** List of prediction dictionaries

**Example:**
```python
path = predictor.predict_path(df, steps=7)
for pred in path:
    print(pred['predicted_price'])
```

### find_support_resistance(df)

Find support and resistance levels.

**Parameters:**
- `df` (pandas.DataFrame): DataFrame with OHLCV data

**Returns:** Dict with support/resistance levels

**Example:**
```python
sr = predictor.find_support_resistance(df)
print(f"Support: {sr['support']}, Resistance: {sr['resistance']}")
```

### generate_trading_signal(symbol, df)

Generate comprehensive trading signal.

**Parameters:**
- `symbol` (str): Cryptocurrency symbol
- `df` (pandas.DataFrame): DataFrame with OHLCV data

**Returns:** Dict with trading signal including entry, exit, stop loss points

**Example:**
```python
signal = predictor.generate_trading_signal('BTC', df)
print(signal['recommendation'])  # BUY, SELL, HOLD, etc.
```

## TechnicalAnalyzer

### find_support_resistance(df, period)

Find support and resistance levels.

**Parameters:**
- `df` (pandas.DataFrame): DataFrame with OHLCV data
- `period` (int): Period for finding extrema (default: 20)

**Returns:** Dict with S/R levels

### calculate_rsi(df, period)

Calculate Relative Strength Index.

**Parameters:**
- `df` (pandas.DataFrame): DataFrame with close prices
- `period` (int): RSI period (default: 14)

**Returns:** pandas.Series with RSI values

### calculate_macd(df)

Calculate MACD indicator.

**Parameters:**
- `df` (pandas.DataFrame): DataFrame with close prices

**Returns:** Dict with MACD, signal line, histogram

### detect_divergence(df)

Detect RSI divergence patterns.

**Parameters:**
- `df` (pandas.DataFrame): DataFrame with OHLCV data

**Returns:** Dict with divergence information

### detect_breakout(df, period)

Detect support/resistance breakouts.

**Parameters:**
- `df` (pandas.DataFrame): DataFrame with OHLCV data
- `period` (int): Period for breakout detection (default: 20)

**Returns:** Dict with breakout information

### get_signals(df)

Generate comprehensive technical signals.

**Parameters:**
- `df` (pandas.DataFrame): DataFrame with OHLCV data

**Returns:** Dict with all technical signals (RSI, MACD, trend, etc.)

## ModelTrainer

### train(X_train, y_train, X_val, y_val, epochs, batch_size, early_stopping_patience)

Train the model.

**Parameters:**
- `X_train` (torch.Tensor): Training features
- `y_train` (torch.Tensor): Training targets
- `X_val` (torch.Tensor): Validation features
- `y_val` (torch.Tensor): Validation targets
- `epochs` (int): Number of training epochs (default: 100)
- `batch_size` (int): Batch size (default: 32)
- `early_stopping_patience` (int): Early stopping patience (default: 10)

**Returns:** Dict with training history

### save_model(path)

Save trained model to disk.

**Parameters:**
- `path` (str): Path to save model

### load_model(path, input_size)

Load trained model from disk.

**Parameters:**
- `path` (str): Path to model file
- `input_size` (int): Number of input features

## DiscordBot

### send_signal_notification(signal)

Send trading signal as Discord embed message.

**Parameters:**
- `signal` (dict): Trading signal dictionary

**Example:**
```python
await discord_bot_cog.send_signal_notification(signal)
```

### send_price_alert(symbol, current_price, alert_type, data)

Send price alert to Discord.

**Parameters:**
- `symbol` (str): Cryptocurrency symbol
- `current_price` (float): Current price
- `alert_type` (str): Type of alert
- `data` (dict): Additional data

### send_error_notification(error_message)

Send error notification to Discord.

**Parameters:**
- `error_message` (str): Error message

### send_summary(crypto_data)

Send market summary to Discord.

**Parameters:**
- `crypto_data` (list): List of cryptocurrency data dictionaries
