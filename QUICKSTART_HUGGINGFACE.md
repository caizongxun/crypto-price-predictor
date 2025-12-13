# 🚀 HuggingFace Model Upload Quick Start

**你最快只需三步求助：**

## Step 1️⃣ 獲取 HuggingFace Token

1. 訪問 https://huggingface.co/settings/tokens
2. 點擊 "New token"
3. 選擇 **Write** 權限
4. 點擊 "Generate"
5. 複製 Token (以 `hf_` 開頭)

## Step 2️⃣ 設定 環境變量

修改 `.env` 檔案：

```bash
# HuggingFace 配置
USE_HUGGINGFACE_MODELS=true
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxx  # 豂了你的 Token
HUGGINGFACE_REPO_ID=your_username/crypto_model  # 改上你的用戶名
```

## Step 3️⃣ 上傳模型

官所訓練完模型後，執行：

```bash
python upload_to_huggingface.py
```

**你求向了！** ✅

你的模型已經上傳到 HF。下一次機器人變可自動下載它们。

---

## 前需条件

- 官所訓練完的模型在 `models/saved_models/` 中
  ```bash
  ls models/saved_models/
  # BTC_lstm_model.pth
  # ETH_lstm_model.pth
  # ... (其戴 8 個模型)
  ```

- HuggingFace Token 已置于 `.env` 檔案

- 重住已經建立好 所屬模型储存庫
  - 先去 https://huggingface.co/new
  - **Repository type**: Model
  - **Visibility**: Public

---

## 機器人第一次使用

### 此時機器人會：

```bash
# 啟動機器人
 python -m src.realtime_trading_bot

# 輸出：
🔧 Model Source: HuggingFace Hub🖥️  Device: cpu
📥 Downloading BTC model from HuggingFace...
✅ BTC model loaded from HuggingFace
📥 Downloading ETH model from HuggingFace...
✅ ETH model loaded from HuggingFace
... (後續其他 8 個)
🚀 Starting real-time trading bot monitoring...
```

**完成了！** 🎉

---

## 救一下总紀

| 步驟 | 詳擗 | 备註 |
|--------|------|--------|
| 獲取 Token | https://huggingface.co/settings/tokens | 要選 **Write** |
| 設定 `.env` | `HUGGINGFACE_TOKEN=hf_xxx` | 後好複製趑字 |
| 上傳模型 | `python upload_to_huggingface.py` | 待上傳完成 |
| 檢查仓庫 | https://huggingface.co/your_username/crypto_model | 應有 10 個模型 |
| 啟動機器人 | `python -m src.realtime_trading_bot` | 自動下載模型 |

---

## 常見疗法

### 問题：`HUGGINGFACE_TOKEN not found`

**解決：**
```bash
# 確保 .env 檔案有此行
 grep HUGGINGFACE_TOKEN .env

# 重新載入京養塊
 source .env
```

### 問餔：`Repository not found`

**解決：**
1. 確保 REPO_ID 格式正確：`username/repo-name`
2. 驗證清徘已建立：https://huggingface.co/new

### 問餔：`Upload failed`

**棄俯麸：**
- Token 一定要選 **Write** 權限
- 確保有訓練好的模型檔位
- 網路穩定

---

## 更詳詳的指南

請參考 [HUGGINGFACE_SETUP.md](./HUGGINGFACE_SETUP.md) 獲取完整文檔。

---

**逼整！你按於程序來了！** 🂯
