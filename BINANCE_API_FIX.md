# 🔧 Binance API 問題修復指南

## ❌ 當前錯誤

```
Failed to fetch SOL/USDT from Binance: binance {"code":-2008,"msg":"Invalid Api-Key ID."}
```

這表示：**API 密鑰無效或沒有正確讀取**

---

## 🔍 診斷步驟

### Step 1: 檢查 .env 文件

1. **打開 PyCharm 根目錄下的 `.env` 文件**
2. 檢查是否有這些行：

```env
BINANCE_API_KEY=your_actual_key_here
BINANCE_API_SECRET=your_actual_secret_here
COINGECKO_API_KEY=optional
DISCORD_WEBHOOK_URL=optional
```

**⚠️ 重要檢查點**：
- ❌ **不要有引號**：`BINANCE_API_KEY="abc123"` ❌ 錯誤
- ✅ **正確格式**：`BINANCE_API_KEY=abc123` ✅ 正確
- ❌ **不要有空格**：`BINANCE_API_KEY = abc123` ❌ 錯誤
- ✅ **正確格式**：`BINANCE_API_KEY=abc123` ✅ 正確

### Step 2: 測試 API 密鑰讀取

在 PyCharm Terminal 中執行：

```python
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('API Key:', os.getenv('BINANCE_API_KEY'))"
```

**預期輸出**:
```
API Key: your_actual_key_xyz123...
```

如果看到 `API Key: None`，說明 `.env` 沒有被正確讀取。

### Step 3: 驗證 API 密鑰的有效性

1. 登錄 [Binance 官方網站](https://www.binance.com/)
2. 打開 API Management（在右上角帳號設置中）
3. 檢查：
   - ✅ API Key 是否還在（沒有被刪除）
   - ✅ API Key 是否被啟用
   - ✅ 是否配置了「IP 白名單」（如果有設置，確保本機 IP 在列表中）
   - ✅ Secret Key 是否正確複製（沒有多餘空格）
   - ✅ API 權限是否包含「讀取」權限

---

## ✅ 解決方案

### 方案 A: 重新創建 API 密鑰（推薦）

如果你的 API 密鑰有問題，最安全的方式是重新生成：

1. **登錄 Binance**
2. **進入 API Management**
3. **刪除舊的 API Key**
4. **創建新的 API Key**：
   - 名稱：例如 `Crypto-Predictor`
   - 限制類型：選擇 `Restrict to IP`
   - IP：填入你本機的 IP（可以在 Binance 中自動檢測）
   - 權限：只勾選 `Read`（只需要讀取數據，不需要交易權限）
5. **複製 API Key 和 Secret Key**

### 方案 B: 修複 .env 文件

1. 在 PyCharm 中打開 `.env`
2. 確保格式完全正確：

```env
# Binance API Configuration
BINANCE_API_KEY=your_api_key_without_quotes_or_spaces
BINANCE_API_SECRET=your_secret_key_without_quotes_or_spaces

# Optional APIs
COINGECKO_API_KEY=optional
DISCORD_WEBHOOK_URL=optional
```

3. **保存文件** (Ctrl + S)
4. **重啟 PyCharm**（這樣才能重新讀取 .env）
   - File → Invalidate Caches → Invalidate and Restart

### 方案 C: 暫時跳過 Binance API，改用備用數據源

如果暫時無法修復 API，可以用其他數據源進行訓練。我已經為你準備了一個無需 API 的版本：

```python
python train_model_ultimate.py --symbol SOL --epochs 100 --use-fallback
```

這會使用 yfinance 或 Kraken 作為備用數據源。

---

## 🛡️ 常見的 Binance API 錯誤代碼

| 錯誤代碼 | 錯誤信息 | 原因 | 解決方案 |
|---------|---------|------|----------|
| **-2008** | Invalid Api-Key ID | API Key 無效或被刪除 | 重新創建 API Key |
| **-1022** | Signature for this request is not valid | Secret Key 錯誤或格式不對 | 檢查 Secret Key 是否正確複製 |
| **-1015** | Too many requests | 請求過於頻繁 | 減少請求頻率或升級 API 權限 |
| **-2015** | Invalid API-key, IP, or permissions | IP 白名單限制 | 檢查本機 IP 是否在白名單中 |
| **-1001** | Mandatory parameter 'symbol' was not sent | 交易對格式錯誤 | 確保使用正確格式（如 SOL/USDT） |

---

## 🔐 安全提示

⚠️ **永遠不要**：

- ❌ 把 API Key 和 Secret 上傳到 GitHub
- ❌ 在代碼中硬編碼 API 密鑰
- ❌ 在公開論壇/截圖中分享 API 密鑰
- ❌ 給 API Key 過度權限（只勾選「Read」就夠了）

✅ **應該做**：

- ✅ 把 API 信息存在 `.env` 文件中
- ✅ 在 `.gitignore` 中排除 `.env`
- ✅ 只給 API 最小必要權限（Read Only）
- ✅ 定期檢查 API 使用情況
- ✅ 如果懷疑洩露，立即刪除該 API Key

---

## 🧪 快速測試

### 測試 1: 驗證 API 密鑰讀取

在 PyCharm Terminal 中執行：

```bash
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')
print(f'API Key Loaded: {bool(api_key)}')
print(f'API Secret Loaded: {bool(api_secret)}')
if api_key:
    print(f'API Key (first 10 chars): {api_key[:10]}...')
"
```

**預期輸出**:
```
API Key Loaded: True
API Secret Loaded: True
API Key (first 10 chars): abc123xyz4...
```

### 測試 2: 驗證 Binance 連接

```bash
python -c "
import ccxt
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')

try:
    binance = ccxt.binance({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True
    })
    ticker = binance.fetch_ticker('SOL/USDT')
    print(f'✅ Binance Connection Success!')
    print(f'SOL/USDT Current Price: ${ticker[\"last\"]:.2f}')
except Exception as e:
    print(f'❌ Binance Connection Failed: {e}')
"
```

如果看到 `✅ Binance Connection Success!` 和當前價格，說明 API 連接正常。

---

## 📝 完整排查清單

### 第 1 步：檢查 .env 文件
- [ ] .env 存在於項目根目錄
- [ ] BINANCE_API_KEY 行沒有引號
- [ ] BINANCE_API_SECRET 行沒有引號
- [ ] 沒有多餘的空格
- [ ] 沒有遺漏等號

### 第 2 步：驗證 API 密鑰
- [ ] 登錄 Binance 官方網站
- [ ] API Management 中 API Key 仍然存在（未被刪除）
- [ ] API Key 已啟用（Enable 狀態）
- [ ] Secret Key 完整複製（沒有截斷）
- [ ] 沒有意外的引號或空格

### 第 3 步：檢查 IP 白名單
- [ ] 如果設置了 IP 限制，確保本機 IP 在列表中
- [ ] 如果不確定本機 IP，可以暫時移除 IP 限制進行測試

### 第 4 步：重啟 PyCharm
- [ ] File → Invalidate Caches → Invalidate and Restart
- [ ] 等候 PyCharm 重新啟動和索引完成

### 第 5 步：重新運行訓練
- [ ] 在新的 PyCharm Terminal 中運行：
  ```bash
  python train_model_ultimate.py --symbol SOL --epochs 100
  ```

---

## 🚀 成功標誌

如果看到這樣的輸出，說明 API 連接成功：

```
2025-12-13 12:53:31 - src.data_fetcher - INFO - Binance API initialized
[Step 1/5] Fetching historical data...
2025-12-13 12:53:32 - src.data_fetcher - INFO - Fetched 500 candles for SOL/USDT
[OK] Fetched 500 candles for SOL/USDT
```

---

## 🆘 還是不行？

如果按照以上步驟還是無法連接 Binance API，嘗試這些：

### 選項 1：使用 yfinance（不需要 API 密鑰）

編輯 `src/data_fetcher.py`，在 `fetch_ohlcv_binance` 前面添加：

```python
def fetch_ohlcv_yfinance_crypto(self, symbol: str, period: str = '1y'):
    """Fallback to yfinance if Binance fails"""
    crypto_symbol = symbol.replace('/USDT', '-USD')
    try:
        ticker = yf.Ticker(crypto_symbol)
        df = ticker.history(period=period, interval='1d')
        df.columns = df.columns.str.lower()
        return df
    except Exception as e:
        logger.error(f"yfinance failed: {e}")
        return None
```

### 選項 2：手動下載歷史數據

如果 API 完全無法工作，可以手動下載 CSV 文件，然後：

```python
df = pd.read_csv('SOL_historical_data.csv', parse_dates=['timestamp'], index_col='timestamp')
```

### 選項 3：使用 CoinGecko API（免費，無需密鑰）

```python
import requests

def fetch_from_coingecko(crypto_id: str, days: int = 365):
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart"
    params = {
        'vs_currency': 'usd',
        'days': days,
        'interval': 'daily'
    }
    response = requests.get(url, params=params)
    return response.json()
```

---

**最後更新**: 2025-12-13  
**版本**: API Fix Guide v1.0
