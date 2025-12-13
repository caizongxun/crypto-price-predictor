# Train all symbols with TFT V3 v1.2 optimization
# Run: .\train_all_multistep.ps1

$symbols = @(
    "BTC",
    "ETH",
    "SOL",
    "DOGE",
    "ADA",
    "XRP",
    "AVAX",
    "MATIC",
    "LINK",
    "ATOM"
)

$epochs = 100
$batch_size = 32
$hidden_size = 256
$num_layers = 3
$lr = 0.001
$forecast_steps = 5
$lookback = 60

Write-Host "================================" -ForegroundColor Cyan
Write-Host "TFT V3 v1.2 Multi-Symbol Trainer" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

$start_time = Get-Date
Write-Host "Start Time: $start_time" -ForegroundColor Yellow
Write-Host "Total Symbols: $($symbols.Count)" -ForegroundColor Yellow
Write-Host "Estimated Time: $($symbols.Count * 15) minutes" -ForegroundColor Yellow
Write-Host ""

foreach ($symbol in $symbols) {
    Write-Host "" -ForegroundColor Green
    Write-Host "======================================" -ForegroundColor Green
    Write-Host "Training: $symbol" -ForegroundColor Green
    Write-Host "======================================" -ForegroundColor Green
    Write-Host "Time: $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Gray
    Write-Host ""
    
    # Run training
    python train_tft_v3_multistep.py `
        --symbol $symbol `
        --epochs $epochs `
        --batch-size $batch_size `
        --hidden-size $hidden_size `
        --num-layers $num_layers `
        --lr $lr `
        --forecast-steps $forecast_steps `
        --lookback $lookback
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "[OK] $symbol training completed" -ForegroundColor Green
        
        # Visualize results
        Write-Host "Generating visualization for $symbol..." -ForegroundColor Cyan
        python visualize_tft_v3_optimized.py `
            --symbol $symbol `
            --lookback $lookback `
            --steps $forecast_steps
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] $symbol visualization generated" -ForegroundColor Green
        } else {
            Write-Host "[WARNING] $symbol visualization failed" -ForegroundColor Yellow
        }
    } else {
        Write-Host "[ERROR] $symbol training failed" -ForegroundColor Red
    }
    
    Write-Host ""
}

Write-Host "" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host "Training Complete!" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

$end_time = Get-Date
$duration = $end_time - $start_time

Write-Host "Start Time: $start_time" -ForegroundColor Yellow
Write-Host "End Time: $end_time" -ForegroundColor Yellow
Write-Host "Total Duration: $($duration.Hours)h $($duration.Minutes)m $($duration.Seconds)s" -ForegroundColor Yellow
Write-Host ""

Write-Host "Training Summary:" -ForegroundColor Cyan
Write-Host "  Models saved to: models/saved_models/" -ForegroundColor Gray
Write-Host "  Metrics saved to: models/training_logs/" -ForegroundColor Gray
Write-Host "  Plots saved to: analysis_plots/" -ForegroundColor Gray
Write-Host ""

Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "  1. Check metrics: cat models/training_logs/*.json" -ForegroundColor Gray
Write-Host "  2. View plots: analysis_plots/" -ForegroundColor Gray
Write-Host "  3. Compare symbols based on MAE/MAPE" -ForegroundColor Gray
Write-Host "  4. Fine-tune best performing models" -ForegroundColor Gray
Write-Host ""
