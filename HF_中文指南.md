# 🚀 HuggingFace 模型上傳中文速查

## 為什麼要用 HuggingFace?

✅ **好處：**
- GitHub 倉庫保持**小體積**（沒有 100MB 的模型檔）
- 模型**自動下載**到機器人
- 免費 **90GB** 存儲空間
- **分版本**管理模型
- 容易和別人**分享**

---

## ⚡ 3 步完成

### 第 1 步：獲取 Token

1. 訪問 https://huggingface.co/settings/tokens
2. 點「New token」
3. **重要**：選 **Write** 權限（不是 Read）
4. 點「Generate」
5. 複製 Token（以 `hf_` 開頭）

### 第 2 步：設置 .env 檔

編輯 `.env` 檔案，加入：

```bash
# HuggingFace 設置
USE_HUGGINGFACE_MODELS=true
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxx          # 貼上你的 Token
HUGGINGFACE_REPO_ID=你的用戶名/crypto_model  # 改成你的用戶名
```

例子：
```bash
HUGGINGFACE_TOKEN=hf_gWeFBl4dJzLkHdUmTrJ9xvM2KpQrStUwNm
HUGGINGFACE_REPO_ID=zongowo111/crypto_model
```

### 第 3 步：上傳模型

官方訓練完模型後，直接執行：

```bash
python upload_to_huggingface.py
```

輸出例子：
```
🏗️  Target Repository: zongowo111/crypto_model
🧠 Found 10 trained models:
  - BTC_lstm_model.pth (4.32 MB)
  - ETH_lstm_model.pth (4.32 MB)
  ... (8 more)

🚀 Starting upload to zongowo111/crypto_model...
✅ Uploaded BTC_lstm_model.pth
✅ Uploaded ETH_lstm_model.pth
...

✅ Upload successful!
🔗 View your repository: https://huggingface.co/zongowo111/crypto_model
```

**完成！** 🎉

---

## 🤖 機器人第一次啟動

現在啟動機器人：

```bash
python -m src.realtime_trading_bot
```

機器人會：
1. ✅ 檢查 `USE_HUGGINGFACE_MODELS=true`
2. 📥 自動從 HuggingFace 下載所有 10 個模型
3. 💾 在本地快取 `~/.cache/huggingface/hub/`
4. ⚡ 下次啟動速度快（直接用快取）

輸出例子：
```
🔧 Model Source: HuggingFace Hub
📥 Downloading BTC model from HuggingFace...
✅ BTC model loaded from HuggingFace
📥 Downloading ETH model from HuggingFace...
✅ ETH model loaded from HuggingFace
... (其他 8 個)
🚀 Starting real-time trading bot monitoring...
```

---

## 📝 常見問題

### Q: 我的 HuggingFace 用戶名在哪？
**A:** 登入後，右上角有你的頭像，點一下就看到用戶名。或訪問 https://huggingface.co/settings/account

### Q: Token 怎麼選 Write 權限？
**A:** 生成 Token 時，下方有 Permissions 選項，選 **Write**

### Q: 模型有多大？
**A:** 每個 ~4.3MB，10 個共 ~43MB。HF 給你 90GB 免費空間。

### Q: 如果不用 HuggingFace 行不行？
**A:** 行！改 `.env`：
```bash
USE_HUGGINGFACE_MODELS=false
```
這樣機器人就用本地模型。

### Q: 重新訓練後可以重新上傳嗎？
**A:** 可以！再執行一次 `python upload_to_huggingface.py` 就會覆蓋舊版本。

### Q: 上傳要多久？
**A:** ~1-2 分鐘（取決於網速）

### Q: 我的模型會不會洩露？
**A:** HuggingFace 默認是公開的。如果怕洩露，可以改成私密：
1. 訪問你的 HF 倉庫 https://huggingface.co/你的用戶名/crypto_model
2. Settings → Visibility → Private
3. 需要別人存取時，給他們 read 權限

---

## 🛠️ 出問題怎麼辦？

### 問題：`HUGGINGFACE_TOKEN not found`

**解決：**
1. 檢查 `.env` 有沒有這行
2. 確認 Token 複製正確（要包含 `hf_`）
3. 如果改了 `.env`，重新執行指令

### 問題：`Repository not found`

**檢查清單：**
- [ ] `HUGGINGFACE_REPO_ID` 格式是 `用戶名/倉庫名` 嗎？
- [ ] 倉庫是否真的存在？訪問 https://huggingface.co/你的用戶名/crypto_model
- [ ] Token 有 Write 權限嗎？

### 問題：上傳失敗

**檢查清單：**
- [ ] Token 有沒有 **Write** 權限？
- [ ] 倉庫存在且公開嗎？
- [ ] `models/saved_models/` 有訓練好的模型嗎？
- [ ] 網路穩定嗎？
- [ ] 模型檔可讀嗎？`ls -la models/saved_models/`

### 問題：下載很慢

**建議：**
- 模型會自動快取，只有第一次慢
- 用有線網路（WiFi 有時候慢）
- 檢查網路：`ping huggingface.co`

---

## 📋 環境變數速查

| 變數 | 說明 | 例子 |
|------|------|------|
| `USE_HUGGINGFACE_MODELS` | 啟用 HF (true/false) | `true` |
| `HUGGINGFACE_TOKEN` | 你的寫入 Token | `hf_xxxxx` |
| `HUGGINGFACE_REPO_ID` | 倉庫 ID | `zongowo111/crypto_model` |

---

## 🔍 檢查上傳是否成功

```bash
# 方法 1：看網頁
訪問 https://huggingface.co/你的用戶名/crypto_model
應該看到 10 個 .pth 檔

# 方法 2：用 Python
python
>>> from src.huggingface_model_manager import HuggingFaceModelManager
>>> manager = HuggingFaceModelManager(repo_id="你的用戶名/crypto_model")
>>> info = manager.get_model_info()
>>> print(info)
# 應該看到 files_count: 10
```

---

## 💾 檔案位置

```
crypto-price-predictor/
├── upload_to_huggingface.py    # ⭐ 上傳腳本
├── HF_中文指南.md              # ⭐ 這個檔
├── QUICKSTART_HUGGINGFACE.md   # 英文快速指南
├── HUGGINGFACE_SETUP.md        # 完整英文教程
├── .env                        # ⭐ 改這個！
├── src/
│   ├── huggingface_model_manager.py
│   ├── realtime_trading_bot.py
│   └── ...
└── models/
    └── saved_models/           # 訓練好的模型在這
        ├── BTC_lstm_model.pth
        ├── ETH_lstm_model.pth
        └── ...
```

---

## ✅ 完整步驟清單

- [ ] 讀完這個檔
- [ ] 訪問 https://huggingface.co/settings/tokens
- [ ] 建立 Token，選 **Write**，複製
- [ ] 開啟 `.env` 檔
- [ ] 填入 `HUGGINGFACE_TOKEN=hf_xxxxx`
- [ ] 填入 `HUGGINGFACE_REPO_ID=你的用戶名/crypto_model`
- [ ] 執行 `python upload_to_huggingface.py`
- [ ] 等待上傳完成
- [ ] 訪問 https://huggingface.co/你的用戶名/crypto_model 驗證
- [ ] 執行 `python -m src.realtime_trading_bot` 測試
- [ ] Git 提交（注意：模型檔不要上傳到 GitHub！）

**總耗時：約 15 分鐘** ⏱️

---

## 🎓 進階用法

### 自己選擇機器人用本地還是 HF 模型

```python
from src.realtime_trading_bot import RealtimeTradingBot

# 用 HuggingFace
bot1 = RealtimeTradingBot(use_huggingface=True)

# 用本地
bot2 = RealtimeTradingBot(use_huggingface=False)
```

### 手動下載單個模型

```python
from src.huggingface_model_manager import HuggingFaceModelManager
import torch

manager = HuggingFaceModelManager(repo_id="你的用戶名/crypto_model")
model = manager.load_model_from_hf(
    symbol="BTC",
    device=torch.device('cpu')
)
print("✅ BTC 模型已加載")
```

---

## 🔗 有用的連結

- **我的 HuggingFace**: https://huggingface.co/settings/profile
- **建立新倉庫**: https://huggingface.co/new
- **Token 管理**: https://huggingface.co/settings/tokens
- **HF 官方文檔**: https://huggingface.co/docs

---

## 💡 提示

💡 **要讓別人用你的模型？**
只需要分享你的倉庫 URL 或用戶名，他們改 `.env` 的 `HUGGINGFACE_REPO_ID` 就行了！

💡 **模型太多怎麼辦？**
HuggingFace 給 90GB 免費空間，足夠存幾千個模型了。

💡 **想要模型備份？**
用 `git lfs` 把 HF 倉庫 clone 下來，就是完整備份。

---

**準備好了嗎？開始上傳吧！** 🎉
