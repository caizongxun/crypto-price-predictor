# 終極批量訓練所有加密貨幣
# Ultimate batch training for maximum accuracy
# 支持 500+ epochs 的長期訓練

param(
    [int]$epochs = 300,           # 推薦 200-500
    [int]$batchSize = 16,         # 小 batch size = 更穩定的訓練
    [string]$device = 'auto',     # auto/cuda/cpu
    [bool]$ultraMode = $true      # 超級模式：更長的訓練時間
)

# 定義加密貨幣
$symbols = @(
    "BTC",    # Bitcoin
    "ETH",    # Ethereum
    "BNB",    # Binance Coin
    "SOL",    # Solana
    "XRP",    # Ripple
    "ADA",    # Cardano
    "DOGE",   # Dogecoin
    "DOT",    # Polkadot
    "AVAX",   # Avalanche
    "MATIC",  # Polygon
    "LTC",    # Litecoin
    "LINK",   # Chainlink
    "UNI",    # Uniswap
    "AAVE",   # Aave
    "COMP"    # Compound
)

# 激活虛擬環境
Write-Host "啟動虛擬環境..." -ForegroundColor Cyan
if (Test-Path ".venv\Scripts\Activate.ps1") {
    . .venv\Scripts\Activate.ps1
} elseif (Test-Path "venv\Scripts\Activate.ps1") {
    . venv\Scripts\Activate.ps1
} else {
    Write-Host "警告：找不到虛擬環境激活腳本" -ForegroundColor Yellow
}

# 記錄開始時間
$startTime = Get-Date
$totalSymbols = $symbols.Count
$completedCount = 0
$failedCount = 0
$failedSymbols = @()

# 超級模式配置
if ($ultraMode) {
    $epochs = [math]::Max($epochs, 300)
    $batchSize = 16
    Write-Host "
⚡ 超級模式啟動！" -ForegroundColor Yellow
    Write-Host "   - 最少 epochs: 300" -ForegroundColor Yellow
    Write-Host "   - Batch Size: 16 (最小化)" -ForegroundColor Yellow
    Write-Host "   - 目標: 最高精度" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "===============================================================================" -ForegroundColor Green
Write-Host "🚀 終極批量訓練 - 最大精度優化" -ForegroundColor Green
Write-Host "===============================================================================" -ForegroundColor Green
Write-Host "開始時間: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Green
Write-Host "訓練配置:" -ForegroundColor Green
Write-Host "  - 模型: Ultimate Ensemble (LSTM-5 + GRU-5 + Transformer-4)" -ForegroundColor Green
Write-Host "  - 隱層大小: 512 | 參數數: ~8.5M" -ForegroundColor Green
Write-Host "  - Epochs: $epochs" -ForegroundColor Green
Write-Host "  - Batch Size: $batchSize (更小 = 更穩定)" -ForegroundColor Green
Write-Host "  - Dropout: 0.6 | L2 Weight Decay: 1e-3" -ForegroundColor Green
Write-Host "  - 設備: $device" -ForegroundColor Green
Write-Host ""

# 預估訓練時間
$estimatedTimePerSymbol = $epochs * 0.5  # 粗略估計
$estimatedTotalTime = $estimatedTimePerSymbol * $totalSymbols / 60
Write-Host "⏱️  預估訓練時間: $([math]::Round($estimatedTotalTime, 1)) 小時" -ForegroundColor Yellow
Write-Host "===============================================================================" -ForegroundColor Green

# 逐個訓練
foreach ($symbol in $symbols) {
    $current = $completedCount + $failedCount + 1
    Write-Host ""
    Write-Host "[$current/$totalSymbols] 訓練: $symbol" -ForegroundColor Yellow
    Write-Host "開始時間: $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Gray
    Write-Host "===============================================================================" -ForegroundColor Gray
    
    # 執行終極版訓練
    python train_model_ultimate.py `
        --symbol $symbol `
        --epochs $epochs `
        --batch-size $batchSize `
        --device $device
    
    # 檢查訓練是否成功
    if ($LASTEXITCODE -eq 0) {
        $completedCount++
        $elapsed = ((Get-Date) - $startTime).TotalSeconds
        Write-Host "[✓] $symbol 訓練完成 - 總耗時: $([math]::Round($elapsed / 60, 1)) 分鐘" -ForegroundColor Green
    } else {
        $failedCount++
        $failedSymbols += $symbol
        Write-Host "[✗] $symbol 訓練失敗" -ForegroundColor Red
    }
    
    # 暫停 2 秒以避免 API 速率限制
    Start-Sleep -Seconds 2
}

# 最終統計
$endTime = Get-Date
$totalDuration = $endTime - $startTime

Write-Host ""
Write-Host "===============================================================================" -ForegroundColor Green
Write-Host "✅ 終極批量訓練完成！" -ForegroundColor Green
Write-Host "===============================================================================" -ForegroundColor Green
Write-Host "完成時間: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Green
Write-Host "成功訓練: $completedCount/$totalSymbols" -ForegroundColor Green
Write-Host "失敗訓練: $failedCount/$totalSymbols" -ForegroundColor $(if ($failedCount -gt 0) { "Red" } else { "Green" })
Write-Host "總耗時: $($totalDuration.TotalHours.ToString('F1')) 小時 ($($totalDuration.TotalMinutes.ToString('F0')) 分鐘)" -ForegroundColor Green
Write-Host ""
Write-Host "📊 模型統計:" -ForegroundColor Cyan
Write-Host "  - 訓練模型類型: Ultimate Ensemble" -ForegroundColor Gray
Write-Host "  - 每個模型大小: ~100-150 MB" -ForegroundColor Gray
Write-Host "  - 總存儲: ~1.5-2.2 GB" -ForegroundColor Gray
Write-Host ""

if ($failedCount -gt 0) {
    Write-Host "失敗的幣種:" -ForegroundColor Red
    foreach ($failed in $failedSymbols) {
        Write-Host "  - $failed" -ForegroundColor Red
    }
    Write-Host ""
}

Write-Host "訓練模型位置: models/saved_models/" -ForegroundColor Green
Write-Host "===============================================================================" -ForegroundColor Green

# 列出所有訓練的模型
Write-Host ""
Write-Host "🎯 訓練的模型列表:" -ForegroundColor Cyan
if (Test-Path "models/saved_models/*ultimate*.pth") {
    Get-ChildItem models/saved_models/*ultimate*.pth | ForEach-Object { 
        Write-Host "  - $($_.Name) (大小: $(($_.Length / 1MB).ToString('F1')) MB)" -ForegroundColor Gray
    }
    Write-Host ""
    Write-Host "💡 提示: 使用 ultimate 模型以獲得最高精度!" -ForegroundColor Green
} else {
    Write-Host "  (未找到模型文件)" -ForegroundColor Gray
}

Write-Host ""
Write-Host "下一步:" -ForegroundColor Cyan
Write-Host "  1. 運行 compare_models.py 查看準確度改進" -ForegroundColor Gray
Write-Host "  2. 運行 visualize_predictions.py 可視化預測" -ForegroundColor Gray
Write-Host "  3. 運行 Discord Bot 進行實時交易信號" -ForegroundColor Gray
Write-Host "===============================================================================" -ForegroundColor Green
