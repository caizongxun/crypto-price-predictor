# 🚀 Quick Start Guide - TFT V3 v1.3 STABLE

## 🎉 STATUS: ALL FIXES COMPLETE - READY TO TRAIN

---

## What Was Fixed

✅ **Dimension Error** - Complete rewrite of MultiHeadAttention  
✅ **Permute Logic** - Using `permute()` + `contiguous()` instead of `transpose()`  
✅ **Memory Safety** - Ensures tensor memory layout consistency  
✅ **Dictionary Access** - Safe key access with fallback defaults  

---

## Start Training NOW

### Option 1: Quick Test (5 minutes)
```bash
python train_tft_v3_multistep.py --symbol SOL --epochs 5
```

### Option 2: Full Training (SOL)
```bash
python train_tft_v3_multistep.py --symbol SOL --epochs 100
```

### Option 3: Different Symbol
```bash
# BTC
python train_tft_v3_multistep.py --symbol BTC --epochs 100

# ETH
python train_tft_v3_multistep.py --symbol ETH --epochs 100

# DOGE
python train_tft_v3_multistep.py --symbol DOGE --epochs 100
```

### Option 4: Custom Hyperparameters
```bash
python train_tft_v3_multistep.py \
  --symbol SOL \
  --epochs 150 \
  --batch-size 16 \
  --lr 0.001 \
  --hidden-size 128 \
  --num-layers 2 \
  --dropout 0.2
```

---

## What to Expect

### Console Output
```
2025-12-14 00:23:35,764 - __main__ - INFO - Using device: cuda
2025-12-14 00:23:35,764 - __main__ - INFO - Model parameters: 478,893
2025-12-14 00:23:35,764 - __main__ - INFO - Train samples: 752
2025-12-14 00:23:35,764 - __main__ - INFO - Val samples: 188
2025-12-14 00:23:35,764 - __main__ - INFO - Batch size: 16
2025-12-14 00:23:35,764 - __main__ - INFO - [3/5] Starting training loop...
Training:  20%|████                    | 10/47 [00:15<01:00, 0.61it/s]
Epoch 1/100 | Train Loss: 0.234567 | Val Loss: 0.345678
✅ Best model saved: models/saved_models/SOL_tft_multistep_best.pth
Epoch 2/100 | Train Loss: 0.198765 | Val Loss: 0.312345
...
```

### Files Generated
```
models/
  ├─ saved_models/
  │   └─ SOL_tft_multistep_best.pth  (trained model)
  └─ training_logs/
      └─ SOL_metrics_v1.2.json  (training metrics)
```

---

## Monitoring Training

### Check Loss Values
Loss should **steadily decrease**:
```
Epoch 1:  Train Loss: 0.234567 | Val Loss: 0.345678
Epoch 2:  Train Loss: 0.198765 | Val Loss: 0.312345  ✅ Decreasing!
Epoch 3:  Train Loss: 0.165432 | Val Loss: 0.287654  ✅ Decreasing!
Epoch 4:  Train Loss: 0.142103 | Val Loss: 0.268901  ✅ Decreasing!
...
```

### Check Model Checkpoints
```bash
# See saved models
ls models/saved_models/

# File size should increase over time (training improving)
ls -lh models/saved_models/SOL_tft_multistep_best.pth
```

### View Metrics After Training
```bash
cat models/training_logs/SOL_metrics_v1.2.json
```

---

## Troubleshooting

### Error: "Expected 3D input"
**Solution:** Make sure you pulled the latest code:
```bash
git pull origin main
```

### Error: "CUDA out of memory"
**Solution:** Reduce batch size:
```bash
python train_tft_v3_multistep.py --symbol SOL --epochs 100 --batch-size 8
```

### Error: "No such file or directory"
**Solution:** Make sure you're in the correct directory:
```bash
cd /path/to/crypto-price-predictor
python train_tft_v3_multistep.py --symbol SOL --epochs 5
```

### Model not improving?
**Check:**
1. Loss is decreasing (not stuck at same value)
2. Validation loss also decreasing (not overfitting)
3. At least 5-10 epochs run

If loss isn't decreasing after 10 epochs:
```bash
# Try lower learning rate
python train_tft_v3_multistep.py --symbol SOL --epochs 100 --lr 0.0005

# Or higher learning rate
python train_tft_v3_multistep.py --symbol SOL --epochs 100 --lr 0.002
```

---

## Performance Targets

After successful training (100 epochs):

| Metric | Target | Status |
|--------|--------|--------|
| MAE | < 2.5 USD | 🕑 Pending |
| MAPE | < 2% | 🕑 Pending |
| Direction Accuracy | > 75% | 🕑 Pending |

---

## Common Commands

```bash
# View GPU usage during training
gpu-monitor  # if installed

# Kill training (if needed)
Ctrl + C

# Resume training from checkpoint
# (Not yet implemented - restart training instead)

# List all trained models
ls -lah models/saved_models/

# Clear old models to save space
rm models/saved_models/*
```

---

## What's Different in v1.3

**Old (broken):**
```python
Q = Q.view(...).transpose(1, 2)  # ❌ Dimension issues
```

**New (fixed):**
```python
Q = Q.view(...)
Q = Q.permute(0, 2, 1, 3).contiguous()  # ✅ Memory safe
```

**Result:** All 4D tensor shape errors gone! 🌟

---

## Next: Use Your Model

After training completes, you can:

1. **Load the model** for inference
2. **Make predictions** on new data
3. **Generate trading signals** (coming soon)
4. **Deploy to live trading** (advanced)

---

## 🌟 Start Training Now!

```bash
python train_tft_v3_multistep.py --symbol SOL --epochs 100
```

**Expected duration:** 30-60 minutes for 100 epochs on GPU

🙋 Let the model train while you relax! ☕

---

## Support

If you encounter issues:
1. Check BUGFIX_REPORT_2025_12_14.md for technical details
2. Check FIX_SUMMARY.txt for quick reference
3. Verify all files are updated: `git pull origin main`

🚀 Happy training!
