# train_all_batch.ps1
# Batch training script for all 15 cryptocurrencies
# Supports custom epochs and batch-size parameters

param(
    [int]$epochs = 100,
    [int]$batchSize = 64,
    [string]$device = 'auto'
)

# Define cryptocurrencies to train
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

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
.\venv\Scripts\Activate.ps1

# Record start time
$startTime = Get-Date
$totalSymbols = $symbols.Count
$completedCount = 0
$failedCount = 0
$failedSymbols = @()

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "Starting batch training for $totalSymbols cryptocurrencies" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "Start Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Green
Write-Host "Training Parameters:" -ForegroundColor Green
Write-Host "  - Epochs: $epochs" -ForegroundColor Green
Write-Host "  - Batch Size: $batchSize" -ForegroundColor Green
Write-Host "  - Device: $device" -ForegroundColor Green
Write-Host ""

# Loop through each symbol for training
foreach ($symbol in $symbols) {
    $current = $completedCount + $failedCount + 1
    Write-Host ""
    Write-Host "[$current/$totalSymbols] Training: $symbol" -ForegroundColor Yellow
    Write-Host "Start Time: $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Gray
    Write-Host "======================================================================" -ForegroundColor Gray
    
    # Execute training command
    python train_model.py --symbol $symbol --epochs $epochs --batch-size $batchSize --device $device
    
    # Check if training was successful
    if ($LASTEXITCODE -eq 0) {
        $completedCount++
        $elapsed = ((Get-Date) - $startTime).TotalSeconds
        Write-Host "[OK] $symbol training completed - Total time: $($elapsed.ToString('F0')) seconds" -ForegroundColor Green
    } else {
        $failedCount++
        $failedSymbols += $symbol
        Write-Host "[ERROR] $symbol training failed" -ForegroundColor Red
    }
    
    # Pause 1 second to avoid API rate limiting
    Start-Sleep -Seconds 1
}

# Final statistics
$endTime = Get-Date
$totalDuration = $endTime - $startTime

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "Batch training completed!" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "Completion Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Green
Write-Host "Successful Training: $completedCount/$totalSymbols" -ForegroundColor Green
Write-Host "Failed Training: $failedCount/$totalSymbols" -ForegroundColor $(if ($failedCount -gt 0) { "Red" } else { "Green" })
Write-Host "Total Time: $($totalDuration.TotalMinutes.ToString('F1')) minutes ($($totalDuration.TotalSeconds.ToString('F0')) seconds)" -ForegroundColor Green

if ($failedCount -gt 0) {
    Write-Host ""
    Write-Host "Failed symbols:" -ForegroundColor Red
    foreach ($failed in $failedSymbols) {
        Write-Host "  - $failed" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Trained models location: models/saved_models/" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green

# List all trained models
Write-Host ""
Write-Host "Trained models list:" -ForegroundColor Cyan
if (Test-Path "models/saved_models/*.pth") {
    Get-ChildItem models/saved_models/*.pth | ForEach-Object { 
        Write-Host "  - $($_.Name) (Size: $(($_.Length / 1MB).ToString('F2')) MB)" -ForegroundColor Gray
    }
} else {
    Write-Host "  (No model files found)" -ForegroundColor Gray
}
