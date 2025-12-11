# train_all_batch.ps1
# 批量訓練所有 15 種加密貨幣（支援自訂 epochs 和 batch-size）

param(
    [int]$epochs = 100,
    [int]$batchSize = 64,
    [string]$device = 'auto'
)

# 定義要訓練的幣種
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

# 啟動虛擬環境
Write-Host "啟動虛擬環境..." -ForegroundColor Cyan
.\venv\Scripts\Activate.ps1

# 記錄開始時間
$startTime = Get-Date
$totalSymbols = $symbols.Count
$completedCount = 0
$failedCount = 0
$failedSymbols = @()

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "開始批量訓練所有 $totalSymbols 種加密貨幣" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "開始時間: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Green
Write-Host "訓練參數:" -ForegroundColor Green
Write-Host "  - Epochs: $epochs" -ForegroundColor Green
Write-Host "  - Batch Size: $batchSize" -ForegroundColor Green
Write-Host "  - Device: $device" -ForegroundColor Green
Write-Host ""

# 遍歷每個幣種進行訓練
foreach ($symbol in $symbols) {
    $current = $completedCount + $failedCount + 1
    Write-Host ""
    Write-Host "[$current/$totalSymbols] 訓練: $symbol" -ForegroundColor Yellow
    Write-Host "開始時間: $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Gray
    Write-Host "======================================================================" -ForegroundColor Gray
    
    # 執行訓練命令
    python train_model.py --symbol $symbol --epochs $epochs --batch-size $batchSize --device $device
    
    # 檢查訓練是否成功
    if ($LASTEXITCODE -eq 0) {
        $completedCount++
        $elapsed = ((Get-Date) - $startTime).TotalSeconds
        Write-Host "[OK] $symbol 訓練完成 - 總耗時: $($elapsed.ToString('F0')) 秒" -ForegroundColor Green
    } else {
        $failedCount++
        $failedSymbols += $symbol
        Write-Host "[ERROR] $symbol 訓練失敗" -ForegroundColor Red
    }
    
    # 暫停 1 秒，避免 API 速率限制
    Start-Sleep -Seconds 1
}

# 最終統計
$endTime = Get-Date
$totalDuration = $endTime - $startTime

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "批量訓練完成！" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "完成時間: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Green
Write-Host "成功訓練: $completedCount/$totalSymbols" -ForegroundColor Green
Write-Host "失敗訓練: $failedCount/$totalSymbols" -ForegroundColor $(if ($failedCount -gt 0) { "Red" } else { "Green" })
Write-Host "總耗時: $($totalDuration.TotalMinutes.ToString('F1')) 分鐘 ($($totalDuration.TotalSeconds.ToString('F0')) 秒)" -ForegroundColor Green

if ($failedCount -gt 0) {
    Write-Host ""
    Write-Host "失敗的幣種:" -ForegroundColor Red
    foreach ($failed in $failedSymbols) {
        Write-Host "  - $failed" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "已訓練的模型位置: models/saved_models/" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green

# 列出所有訓練好的模型
Write-Host ""
Write-Host "已訓練的模型列表:" -ForegroundColor Cyan
if (Test-Path "models/saved_models/*.pth") {
    Get-ChildItem models/saved_models/*.pth | ForEach-Object { 
        Write-Host "  - $($_.Name) (大小: $(($_.Length / 1MB).ToString('F2')) MB)" -ForegroundColor Gray
    }
} else {
    Write-Host "  (還沒有模型文件)" -ForegroundColor Gray
}
