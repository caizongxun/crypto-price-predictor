# train_all_batch.ps1
# 批量訓練所有 15 種加密貨幣

$symbols = @("BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "DOT", "AVAX", "MATIC", "LTC", "LINK", "UNI", "AAVE", "COMP")

.\venv\Scripts\Activate.ps1

$startTime = Get-Date
$success = 0
$failed = 0

foreach ($symbol in $symbols) {
    Write-Host ""
    Write-Host "===== 訓練: $symbol =====" -ForegroundColor Green

    python train_model.py --symbol $symbol --epochs 100 --batch-size 64

    if ($LASTEXITCODE -eq 0) {
        $success++
        Write-Host "[OK] $symbol 完成" -ForegroundColor Green
    } else {
        $failed++
        Write-Host "[ERROR] $symbol 失敗" -ForegroundColor Red
    }
}

$duration = ((Get-Date) - $startTime).TotalMinutes
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "批量訓練完成！" -ForegroundColor Green
Write-Host "成功: $success, 失敗: $failed" -ForegroundColor Green
Write-Host "總耗時: $($duration.ToString('F1')) 分鐘" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
