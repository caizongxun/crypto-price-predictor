# 📊 TFT V3 可視化工具完整指南

## 🎯 快速導航

| 工具 | 用途 | 時間 | 命令 |
|------|------|------|------|
| `quick_visualize.py` | 快速檢查 | 30秒 | `python quick_visualize.py --symbol SOL` |
| `monitor_training.py` | 實時監控 | 訓練中 | `python monitor_training.py` |
| `visualize_tft_v3.py` | 詳細評估 | 1-2分鐘 | `python visualize_tft_v3.py --symbol SOL` |

---

## 🚀 推薦工作流程

### 第1步：快速檢查（現在就試）
```bash
python quick_visualize.py --symbol SOL
```
**輸出：** `quick_analysis.png` (4張圖) + 性能指標

**檢查內容：**
- ✓ 數據質量
- ✓ 特徵工程
- ✓ 指標計算
- ✓ 準備訓練

### 第2步：開始訓練（新終端）
```bash
python train_tft_v3.py --symbol SOL --epochs 200
```
**時間：** 3-4小時（GPU）/ 12+ 小時（CPU）

### 第3步：實時監控（另一個新終端）
```bash
python monitor_training.py
```
**更新：** 每10秒刷新一次 `training_progress.png`

### 第4步：訓練完成後評估
```bash
python visualize_tft_v3.py --symbol SOL --steps 5
```
**輸出：** 9張詳細分析圖表 + 完整報告

---

## 📊 工具詳細說明

### 1. quick_visualize.py - 快速可視化

#### 功能
- 自動獲取最新價格數據
- 生成樣本預測
- 繪製4個分析圖表
- 計算性能指標

#### 命令
```bash
# 基本用法
python quick_visualize.py --symbol SOL

# 指定輸出文件
python quick_visualize.py --symbol BTC --output btc_analysis.png

# 指定方法版本
python quick_visualize.py --symbol ETH --method v3
```

#### 生成的圖表

**圖表1：價格曲線 (實際 vs 預測)**
- 藍線：實際價格走勢
- 紅點線：模型預測
- 灰色區：預測誤差範圍
- 用途：評估模型追蹤能力

**圖表2：誤差分佈直方圖**
- 紫色柱狀圖：誤差頻率分佈
- 紅色虛線：零誤差基線
- 用途：檢查誤差是否正常分佈

**圖表3：實際 vs 預測散點圖**
- 綠色點：預測樣本
- 紅色虛線：完美預測線
- 用途：直觀評估預測準確性

**圖表4：性能指標表**
- MAE、RMSE、MAPE、R²
- 方向準確度
- 樣本統計信息

#### 輸出文件
```
quick_analysis.png          # 4張子圖組合
```

#### 性能指標說明
| 指標 | 含義 | 目標 |
|------|------|------|
| MAE | 平均絕對誤差 | < 2 USD |
| RMSE | 均方根誤差 | < 2.5 USD |
| MAPE | 平均百分比誤差 | < 2% |
| R² | 決定係數 | > 0.9 |
| 方向準確度 | 漲跌預測正確率 | > 60% |

---

### 2. monitor_training.py - 實時訓練監控

#### 功能
- 實時解析訓練日誌
- 動態繪製損失曲線
- 跟踪學習率變化
- 計算改進百分比
- 自動更新圖表

#### 命令
```bash
# 基本用法（每10秒更新一次）
python monitor_training.py

# 自定義更新頻率
python monitor_training.py --interval 5      # 每5秒更新

# 監控指定時間
python monitor_training.py --duration 120    # 監控2小時

# 指定日誌文件
python monitor_training.py --log logs/custom_log.log

# 一次性生成（訓練完成後）
python monitor_training.py --plot-only
```

#### 生成的圖表

**圖表1：損失曲線**
- 藍線：訓練損失（Train Loss）
- 紅線：驗證損失（Val Loss）
- 用途：監控訓練進度和過擬合

**圖表2：改進百分比**
- 綠色填充區：改進幅度
- 用途：評估優化效果

**圖表3：學習率曲線**
- 紫色曲線：學習率變化
- 對數刻度顯示
- 用途：理解優化策略

**圖表4：統計摘要**
- 當前 Epoch
- Train/Val Loss
- 最佳驗證損失
- 學習率
- 總體改進百分比

#### 輸出文件
```
training_progress.png       # 4張子圖（實時更新）
```

#### 使用場景
- 訓練進行中想檢查進度
- 觀察過擬合傾向
- 確認訓練是否卡頓
- 驗證學習率調度

---

### 3. visualize_tft_v3.py - 完整模型評估

#### 功能
- 加載已訓練的 TFT V3 模型
- 多步預測（3-5根K棒）
- 置信區間計算
- 波動率調整分析
- 特徵重要性評估
- 詳細性能報告

#### 命令
```bash
# 基本用法
python visualize_tft_v3.py --symbol SOL

# 指定預測步數
python visualize_tft_v3.py --symbol SOL --steps 5

# 使用 V2 模型
python visualize_tft_v3.py --symbol BTC --method v2

# 指定輸出目錄
python visualize_tft_v3.py --symbol ETH --output-dir results/
```

#### 生成的圖表（9張）

1. **價格預測曲線** - 完整預測軌跡
2. **多步預測** - 3-5根K棒ahead
3. **置信區間** - 預測不確定性
4. **誤差分佈** - 預測誤差統計
5. **實際 vs 預測** - 散點圖評估
6. **波動率分析** - 波動率調整準確性
7. **特徵重要性** - 各特徵貢獻度
8. **注意力熱圖** - 模型注意力分佈
9. **性能對比表** - 完整指標摘要

#### 輸出文件
```
{SYMBOL}_tft_v3_analysis_{TIMESTAMP}.png    # 9張子圖
{SYMBOL}_predictions.csv                     # 預測結果
{SYMBOL}_metrics.json                        # 詳細指標
{SYMBOL}_performance_report.txt              # 文本報告
```

#### 使用場景
- 訓練完成後的最終評估
- 性能基準測試
- 生產部署前驗證
- 提交分析報告

---

## 💡 關鍵指標詳解

### MAE (Mean Absolute Error)
```
公式：MAE = (1/n) * Σ|y_actual - y_predicted|

含義：
  - 預測平均偏差（單位：USD）
  - 如 MAE=1.5 表示平均誤差 1.5 USD
  - 越小越好

目標：< 2 USD
```

### MAPE (Mean Absolute Percentage Error)
```
公式：MAPE = (1/n) * Σ|((y_actual - y_predicted) / y_actual) * 100%|

含義：
  - 預測百分比誤差
  - 如 MAPE=1.2% 表示平均誤差 1.2%
  - 不受絕對價格影響

目標：< 2%
```

### R² (Coefficient of Determination)
```
公式：R² = 1 - (SS_res / SS_tot)

含義：
  - 模型擬合度
  - 範圍：0 到 1（1 最佳）
  - R²=0.92 表示模型解釋 92% 的價格變動

目標：> 0.9
```

### Directional Accuracy
```
公式：Accuracy = (正確方向預測數 / 總預測數) * 100%

含義：
  - 預測漲跌方向的正確率
  - 不受價格幅度影響
  - 重要的實用指標

目標：> 60%
```

---

## ⚠️ 常見問題與解決方案

### 問題1：快速可視化失敗

**症狀：**
```
Error: Binance API connection failed
```

**原因：**
- 網絡連接問題
- Binance API 限流
- 無效的交易對

**解決：**
1. 檢查網絡連接
2. 稍等後重試
3. 驗證交易對名稱（如 SOL/USDT）

### 問題2：監控工具不更新

**症狀：**
```
Waiting for training to start...
```

**原因：**
- 訓練腳本未啟動
- 日誌文件不存在
- 日誌文件無更新

**解決：**
1. 確保訓練腳本正在運行：`python train_tft_v3.py ...`
2. 檢查日誌文件：`logs/training_tft_v3.log`
3. 嘗試手動更新：`python monitor_training.py --plot-only`

### 問題3：訓練損失不下降

**症狀：**
```
Epoch  10/200 | Train Loss: 0.5234 | Val Loss: 0.5245
Epoch  20/200 | Train Loss: 0.5233 | Val Loss: 0.5244
(無變化)
```

**原因：**
- 學習率設置不合理
- 數據質量問題
- 模型架構不適合

**解決：**
1. 檢查學習率（監控圖表中的學習率曲線）
2. 嘗試調整：`--lr 0.0005` 或 `--lr 0.00005`
3. 增加 epoch 數
4. 檢查數據中是否有異常

### 問題4：驗證損失上升（過擬合）

**症狀：**
```
Epoch  30/200 | Train Loss: 0.001 | Val Loss: 0.050  ← Val 損失變大
```

**原因：**
- 模型在訓練數據上過擬合
- Dropout 或正則化不足
- 訓練時間過長

**解決：**
1. 早期停止（會自動觸發）
2. 增加 Dropout
3. 增加 L2 正則化
4. 使用更多數據增強

---

## 📁 文件組織

```
crypto-price-predictor/
├── quick_visualize.py              # 快速可視化工具
├── monitor_training.py             # 實時監控工具
├── train_tft_v3.py                 # 訓練腳本
├── visualize_tft_v3.py             # 詳細評估工具
│
├── logs/
│   └── training_tft_v3.log         # 訓練日誌
│
├── models/
│   └── saved_models/
│       └── SOL_tft_model.pth       # 已訓練的模型
│
├── quick_analysis.png              # 快速可視化輸出
├── training_progress.png           # 訓練監控輸出
│
└── analysis_plots/
    └── SOL_tft_v3_analysis_*.png   # 詳細分析輸出
```

---

## 🔄 完整工作流程示例

### 終端 1：快速檢查
```bash
$ python quick_visualize.py --symbol SOL
[1/3] Fetching data for SOL...
[2/3] Preparing features...
[3/3] Generating plots...

ANALYSIS COMPLETE
MAE:                 1.2340 USD
MAPE:                0.92%
R²:                  0.9234
Output: quick_analysis.png
```

### 終端 2：開始訓練
```bash
$ python train_tft_v3.py --symbol SOL --epochs 200
2025-12-13 21:32:55 - src.utils - INFO - Logging to file: logs/training_tft_v3.log
2025-12-13 21:32:55 - src.utils - INFO - Created 5 directories

Fetching data for SOL...
Adding indicators...
Preparing features...

================================================================================
TFT V3 TRAINING - ADVANCED OPTIMIZATION
================================================================================

[1/6] Applying advanced data augmentation...
Augmented 921 samples to 3684 samples

[2/6] Splitting data (80/20 train/val)...
  - Train: 2947 samples
  - Val:   737 samples

[3/6] Creating dataloaders...
Dataloaders created

[4/6] Setting up training components...
  Optimizer: AdamW (lr=0.0001, weight_decay=0.001)
  Loss: Combined (MSE + Weighted + Directional)
  Scheduler: Cosine Annealing with Warm Restarts

[5/6] Training for 200 epochs...

Epoch  10/200 | Train Loss: 0.004321 | Val Loss: 0.004567 | LR: 9.97e-05
Epoch  20/200 | Train Loss: 0.003892 | Val Loss: 0.003456 | LR: 9.89e-05
...
```

### 終端 3：實時監控（同時運行）
```bash
$ python monitor_training.py

======================================================================
TFT V3 Training Monitor - 2025-12-13 21:42:15
======================================================================

Epoch: 10 (Total: 10 epochs)

Loss:
  Train: 0.004321
  Val:   0.004567
  Best:  0.004567

Learning Rate: 9.97e-05
Improvement: 8.3%
======================================================================
```

### 訓練完成後：詳細評估
```bash
$ python visualize_tft_v3.py --symbol SOL --steps 5
[1/4] Loading model...
[2/4] Generating predictions...
[3/4] Creating analysis plots...
[4/4] Generating report...

Output: SOL_tft_v3_analysis_2025-12-13_21-45-23.png
Metrics: SOL_metrics.json
```

---

## 🎓 最佳實踐

### ✅ DO
- ✓ 訓練前用 `quick_visualize.py` 檢查數據
- ✓ 訓練中用 `monitor_training.py` 實時監控
- ✓ 訓練後用 `visualize_tft_v3.py` 詳細評估
- ✓ 保存所有圖表用於文檔
- ✓ 定期檢查性能指標

### ❌ DON'T
- ✗ 跳過數據檢查直接訓練
- ✗ 訓練中不監控進度
- ✗ 忽略過擬合警告
- ✗ 不保存訓練日誌
- ✗ 使用硬編碼的超參數

---

## 📞 支持

如有問題，請檢查：
1. 日誌文件：`logs/training_tft_v3.log`
2. GitHub Issues
3. 本指南的常見問題部分

---

**祝你訓練順利！🚀**
