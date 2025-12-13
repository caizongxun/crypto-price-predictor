# 🚀 PyCharm 本地訓練完整指南

## 📋 前置需求檢查

### 1️⃣ 系統環境

- ✅ **Python 版本**: 3.9 或以上
  ```powershell
  python --version
  ```
  
- ✅ **GPU（可選但推薦）**: NVIDIA GPU + CUDA
  ```powershell
  nvidia-smi
  ```
  如果沒有輸出，說明沒有 GPU，訓練會很慢（但仍可用 CPU）

- ✅ **硬碟空間**: 至少 10GB
  - 模型文件: ~2GB (15 個幣種)
  - 數據緩存: ~3GB
  - 日誌文件: ~1GB
  - 預留空間: 安全起見，準備 15GB

### 2️⃣ PyCharm 版本

- 推薦: **PyCharm Professional 或 Community 版本（2022.3+）**
- 下載: [JetBrains PyCharm](https://www.jetbrains.com/pycharm/)

---

## 🔧 Step 1: 在 PyCharm 中打開項目

### 1.1 打開項目

1. **啟動 PyCharm**
2. 選擇 `File` → `Open`
3. 選擇你的 `crypto-price-predictor` 文件夾
4. 點擊 `Open as Project`

```
PyCharm 會自動掃描項目結構
顯示:
  src/
  models/
  logs/
  train_model_ultimate.py
  train_all_ultimate.ps1
  ...
```

### 1.2 配置 Python 解釋器

**這是最重要的一步！**

#### 方法 A: 使用現有虛擬環境（推薦）

```
1. 在 PyCharm 中打開 Settings
   - Windows/Linux: Ctrl + Alt + S
   - Mac: Cmd + ,

2. 導航到: Project → Python Interpreter

3. 點擊右上角 ⚙️ 圖標 → Add

4. 選擇 "Existing Environment"

5. 找到虛擬環境的 Python 可執行文件:
   - Windows: .venv\Scripts\python.exe
              或 venv\Scripts\python.exe
   - Mac/Linux: .venv/bin/python
              或 venv/bin/python

6. 點擊 "OK"
```

**驗證成功**:

```
如果看到:
  ✅ "Python 3.x.x (.venv)" 在 Interpreter 下拉菜單
  ✅ 顯示已安裝的包 (torch, numpy, pandas 等)

說明配置成功！
```

#### 方法 B: 創建新虛擬環境（如果還沒有）

```
1. Settings → Project → Python Interpreter

2. 點擊 ⚙️ 圖標 → Add

3. 選擇 "New Environment"

4. 選擇位置: <Project Path>\.venv 或 <Project Path>\venv

5. 點擊 "Create"

6. PyCharm 會自動創建虛擬環境
   （需要 2-5 分鐘）

7. 創建完後，安裝依賴:
   在 PyCharm Terminal 中執行:
   pip install -r requirements.txt
```

---

## 📦 Step 2: 安裝依賴

### 2.1 打開 PyCharm Terminal

```
View → Tool Windows → Terminal

或快捷鍵:
  Alt + F12 (Windows/Linux)
  Cmd + Alt + F (Mac)
```

### 2.2 檢查虛擬環境是否激活

```powershell
# 應該看到 (.venv) 或 (venv) 前綴
(.venv) PS C:\...\crypto-price-predictor>
```

如果沒有激活，手動激活：

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1

# 或 Windows CMD
venv\Scripts\activate.bat

# Mac/Linux
source venv/bin/activate
```

### 2.3 安裝依賴包

```powershell
# 升級 pip
pip install --upgrade pip

# 安裝所有依賴
pip install -r requirements.txt

# 如果 requirements.txt 丟失，手動安裝關鍵包
pip install torch numpy pandas scikit-learn requests python-dotenv
```

**安裝進度**:

```
預計時間: 5-10 分鐘
（取決於網速和是否要下載 PyTorch）

完成標誌:
✅ Successfully installed ...
```

---

## 🎯 Step 3: 運行訓練

### 3.1 單個幣種訓練（推薦先試這個）

#### 方法 A: 直接運行腳本（最簡單）

**用滑鼠點：**

```
1. 在 PyCharm 左側文件瀏覽器中
   右擊 train_model_ultimate.py

2. 選擇 "Run 'train_model_ultimate'"

3. PyCharm 會自動執行
```

#### 方法 B: 帶參數運行（推薦）

**在 PyCharm Terminal 中：**

```powershell
# 基礎訓練（300 epochs）
python train_model_ultimate.py --symbol SOL --epochs 300

# 完整配置
python train_model_ultimate.py `
    --symbol BTC `
    --epochs 300 `
    --batch-size 16 `
    --learning-rate 0.00005 `
    --device cuda
```

**參數解釋**:

| 參數 | 默認值 | 說明 |
|------|--------|------|
| `--symbol` | BTC | 訓練的幣種 (BTC, ETH, SOL 等) |
| `--epochs` | 300 | 訓練輪次（推薦 300-500） |
| `--batch-size` | 16 | 批次大小（小 = 穩定，慢） |
| `--learning-rate` | 0.00005 | 學習率 |
| `--device` | auto | 設備選擇 (auto, cuda, cpu) |

### 3.2 實時監控訓練

**PyCharm 會在 Run 面板顯示輸出**:

```
[00:05:23] Epoch 10/300 | Train: 0.052341 | Val: 0.078234 | Ratio: 1.495
[00:10:47] Epoch 20/300 | Train: 0.048932 | Val: 0.075123 | Ratio: 1.534
[00:16:12] Epoch 30/300 | Train: 0.045678 | Val: 0.072456 | Ratio: 1.585

✅ 這表示訓練在進行中！
```

**重要指標**:

```
🔴 Ratio > 1.6  → 過擬合太嚴重
🟡 1.4 < Ratio < 1.6 → 可以接受
🟢 Ratio < 1.3  → 很好！
```

### 3.3 查看完整日誌

```
1. PyCharm 下方有 "Run" 面板

2. 點擊 Run 面板中的 "Open in Editor" 按鈕
   或直接打開文件:
   
   logs/training_ultimate.log
```

---

## 🔁 Step 4: 批量訓練所有幣種

### 4.1 在 PyCharm 中運行批量腳本

**方式 A: 通過 Terminal（Windows PowerShell）**

```powershell
# 在 PyCharm Terminal 中
.\train_all_ultimate.ps1 -epochs 300 -batchSize 16
```

**方式 B: 創建自定義 Run Configuration（更專業）**

```
1. 在 PyCharm 中點擊頂部菜單
   Run → Edit Configurations

2. 點擊 + 添加新配置

3. 選擇 "Python"

4. 填寫:
   Name: Ultimate Batch Training
   Script path: train_model_ultimate.py
   Parameters: --symbol BTC --epochs 300 --batch-size 16
   Python interpreter: 選擇你的虛擬環境
   Working directory: <Project Root>

5. 點擊 "Apply" → "OK"

6. 以後可以直接點擊運行按鈕快速執行
```

### 4.2 創建多個訓練配置（高級技巧）

**一鍵切換訓練不同幣種**:

```
1. Run → Edit Configurations

2. 創建多個配置:
   - "Train BTC Ultimate" (--symbol BTC)
   - "Train ETH Ultimate" (--symbol ETH)
   - "Train SOL Ultimate" (--symbol SOL)
   ...

3. 然後在頂部下拉菜單中快速選擇
```

---

## 🎨 Step 5: 設置 PyCharm 調試（可選但推薦）

### 5.1 添加斷點進行調試

```
1. 打開 src/model_trainer_ultimate.py

2. 在某一行左側點擊（例如第 250 行）
   會出現紅色圓點 (斷點)

3. Run → Debug (或 Shift + F9)

4. 執行會在斷點停下
   可以檢查變量、單步執行等
```

### 5.2 查看變量監視

```
調試時，右下角 "Variables" 面板會顯示:
- self.device
- X.shape
- model parameters
...

對於理解訓練流程很有幫助
```

---

## 📊 Step 6: 訓練後評估結果

### 6.1 運行比較腳本

```powershell
# 在 PyCharm Terminal 中
python compare_models.py

# 會輸出:
# BTC: MAE = 0.1125 (改善 33%)
# ETH: MAE = 0.1089 (改善 34%)
# SOL: MAE = 0.1052 (改善 32%)
# ...
```

### 6.2 可視化預測

```powershell
# 生成圖表
python visualize_predictions.py

# 會生成:
# results/BTC_prediction_1h.png
# results/ETH_prediction_1h.png
# ...
```

### 6.3 在 PyCharm 中查看結果圖

```
1. 左側文件瀏覽器中
   展開 results/ 文件夾

2. 右擊任意 .png 文件
   選擇 "Open with" → "Default viewer"

3. 會在新窗口打開圖片
```

---

## ⚙️ Step 7: 配置 Python Console（高級功能）

### 7.1 打開 Python Console

```
View → Tool Windows → Python Console

或快捷鍵:
  Alt + Shift + E (Windows)
```

### 7.2 在 Console 中快速測試

```python
# 例如，快速測試模型加載
import torch
from src.model_trainer_ultimate import UltimateEnsembleModel

# 加載已訓練模型
model = UltimateEnsembleModel(...)
model.load_state_dict(torch.load('models/saved_models/SOL_ultimate_model.pth'))

# 測試推論
with torch.no_grad():
    pred = model(sample_input)
    print(f"Prediction: {pred}")
```

---

## 🐛 Step 8: 常見問題排除

### 問題 1: Python Interpreter 找不到

**症狀**:
```
"Error: No Python Interpreter configured"
```

**解決**:
```
1. Settings → Project → Python Interpreter

2. 點擊 ⚙️ → Add

3. 選擇 "Existing Environment"

4. 手動導航到:
   .venv\Scripts\python.exe
   
   或 venv\Scripts\python.exe

5. 應該能找到
```

### 問題 2: "Module not found" 錯誤

**症狀**:
```
ModuleNotFoundError: No module named 'torch'
```

**解決**:
```powershell
# 在 PyCharm Terminal 中
pip install torch

# 或完整安裝
pip install -r requirements.txt

# 然後在 PyCharm 中:
# File → Invalidate Caches → Invalidate and Restart
```

### 問題 3: GPU 識別不到 (CUDA not available)

**症狀**:
```
Device: CPU (not GPU)
```

**解決**:

```powershell
# 檢查 NVIDIA GPU
nvidia-smi

# 如果沒有輸出，說明:
# 1. 沒有 NVIDIA GPU
# 2. 驅動程序沒有安裝
# 3. CUDA 沒有安裝

# 重新安裝 PyTorch (支持 CUDA):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 或強制使用 CPU (慢但可用):
python train_model_ultimate.py --device cpu
```

### 問題 4: 訓練速度太慢

**原因 1: 使用了 CPU**

```powershell
# 檢查
python -c "import torch; print(torch.cuda.is_available())"

# 如果返回 False，參考問題 3
```

**原因 2: Batch Size 太小**

```powershell
# 試試增大 batch size
python train_model_ultimate.py --batch-size 32

# 但如果 GPU 內存溢出，再改回 16
```

**原因 3: 就是 CPU 訓練**

```
只能等待... 或投資一個 GPU

估計時間:
- GPU (RTX 3080): 30-40 分鐘/幣種
- GPU (RTX 2080): 60-90 分鐘/幣種
- CPU (i7): 8-12 小時/幣種
- CPU (i5): 12-24 小時/幣種
```

### 問題 5: "Permission denied" 或無法寫入日誌

**症狀**:
```
FileNotFoundError: logs/training_ultimate.log
```

**解決**:
```powershell
# 創建 logs 文件夾
mkdir logs
mkdir models
mkdir models/saved_models
mkdir results

# 確保權限
attrib -r logs models results
```

---

## 📈 推薦訓練流程

### Day 1: 測試階段

```bash
# 1. 確保環境配置正確 (30 分鐘)
# 在 PyCharm 中驗證 Python Interpreter

# 2. 訓練 SOL (相對穩定的幣種)
python train_model_ultimate.py --symbol SOL --epochs 100
# 預期耗時: 15-30 分鐘

# 3. 評估結果
python compare_models.py
python visualize_predictions.py

# 4. 檢查是否有明顯改進
# ✅ MAE 降低 20%+ → 進行完整訓練
# ❌ MAE 沒有改進 → 調試參數
```

### Day 2-3: 完整訓練

```bash
# 1. 調整參數到最優（基於 Day 1 結果）

# 2. 訓練最重要的 3 個幣種 (BTC, ETH, SOL)
# 可以在 PyCharm 中同時開 3 個終端，並行訓練

# Terminal 1:
python train_model_ultimate.py --symbol BTC --epochs 300

# Terminal 2:
python train_model_ultimate.py --symbol ETH --epochs 300

# Terminal 3:
python train_model_ultimate.py --symbol SOL --epochs 300

# 預期耗時: 2-3 小時（如果有 GPU）
```

### Day 4: 批量訓練所有幣種

```bash
# 一次性訓練所有 15 個幣種
.\train_all_ultimate.ps1 -epochs 300 -batchSize 16

# 預期耗時: 8-12 小時（單 GPU）
# 可以在訓練期間做其他工作
```

### Day 5: 最終評估

```bash
# 1. 所有訓練完成後
python compare_models.py

# 2. 生成最終可視化
python visualize_predictions.py

# 3. 將結果提交到 Discord Bot 或生產環境
```

---

## 💡 PyCharm 專業技巧

### Tip 1: 用 Shift + F10 快速重新運行上次命令

```
第一次: Run → Edit Configurations → 設置完成
第二次及以後: 只需按 Shift + F10
```

### Tip 2: 在 Run 面板中搜索日誌

```
1. 訓練過程中，Run 面板會顯示大量輸出

2. 使用 Ctrl + F 搜索特定內容
   例如: "Epoch 100" 可以快速找到該 epoch
```

### Tip 3: 設置書籤快速導航

```
1. 打開 train_model_ultimate.py

2. 在某一行點擊
   右擊 → Add Bookmark

3. 以後可以用 Ctrl + 1, 2, 3 快速跳轉
```

### Tip 4: 用 TODO 標記待做項

```python
# 在代碼中添加
# TODO: 這裡需要優化精度
# FIXME: 這裡可能有 bug

# PyCharm 會自動識別
# View → Tool Windows → TODO
```

### Tip 5: 使用版本控制 (Git)

```
1. VCS → Enable Version Control → Git

2. 訓練完成後提交：
   VCS → Commit

3. 推送到 GitHub:
   VCS → Git → Push
```

---

## 🎓 下一步

1. ✅ 完成 Step 1-3: Python 環境配置 + 單幣訓練
2. ✅ 驗證結果是否改進
3. ✅ 完成 Step 4: 批量訓練所有幣種
4. ✅ 完成 Step 6: 評估最終結果
5. ✅ 部署到 Discord Bot 或生產環境

---

## 📞 需要幫助？

如果訓練過程中遇到問題，記錄以下信息：

```
1. Python 版本: python --version
2. PyTorch 版本: python -c "import torch; print(torch.__version__)"
3. 是否有 GPU: nvidia-smi
4. 完整錯誤信息 (複製粘貼最後 20 行日誌)
5. 運行的命令
```

---

**最後更新**: 2025-12-13  
**版本**: PyCharm Local Training Guide v1.0  
**建議環境**: PyCharm 2023.3+, Python 3.9+, NVIDIA GPU (可選)
