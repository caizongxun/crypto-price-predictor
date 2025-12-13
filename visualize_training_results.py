#!/usr/bin/env python3
"""
Visualize Training Results and Model Evaluation

View training metrics with matplotlib charts
Works in PyCharm and command line

Usage:
  python visualize_training_results.py --symbol SOL
  python visualize_training_results.py --symbol SOL --eval
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import torch
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from src.data_fetcher_tft_v3 import TFTDataFetcher
from src.model_tft_v3_optimized import TemporalFusionTransformerV3Optimized


def plot_training_metrics(symbol, model_type='directional'):
    """
    Plot training metrics from logs
    """
    log_file = Path(f'logs/training_tft_v3_{model_type}.log')
    
    if not log_file.exists():
        print(f"Log file not found: {log_file}")
        return
    
    # Parse log file for metrics
    train_losses = []
    val_losses = []
    train_dir_accs = []
    val_dir_accs = []
    epochs = []
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    for line in lines:
        if 'Epoch' in line and 'Loss' in line and symbol in line:
            try:
                parts = line.split('|')
                epoch_part = parts[0].split('Epoch')[1].strip().split('/')[0]
                
                loss_part = parts[1].strip().split('Loss:')[1].strip().split('/')
                train_loss = float(loss_part[0])
                val_loss = float(loss_part[1])
                
                if 'Dir Acc' in line:
                    acc_part = parts[2].strip().split('Dir Acc:')[1].strip().split('/')
                    train_acc = float(acc_part[0].rstrip('%')) / 100
                    val_acc = float(acc_part[1].rstrip('%')) / 100
                    
                    epochs.append(int(epoch_part))
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    train_dir_accs.append(train_acc)
                    val_dir_accs.append(val_acc)
            except:
                continue
    
    if not epochs:
        print("No training metrics found in log")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training Metrics - {symbol} ({model_type.upper()})', fontsize=16, fontweight='bold')
    
    # Plot 1: Training Loss
    ax = axes[0, 0]
    ax.plot(epochs, train_losses, label='Train Loss', marker='o', markersize=4, linewidth=2, color='blue')
    ax.plot(epochs, val_losses, label='Val Loss', marker='s', markersize=4, linewidth=2, color='red')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Loss Evolution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    if train_losses and val_losses:
        ax.set_yscale('log')
    
    # Plot 2: Directional Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, [acc*100 for acc in train_dir_accs], label='Train Dir Acc', marker='o', markersize=4, linewidth=2, color='green')
    ax.plot(epochs, [acc*100 for acc in val_dir_accs], label='Val Dir Acc', marker='s', markersize=4, linewidth=2, color='orange')
    ax.axhline(y=33.33, color='r', linestyle='--', label='Random (33%)', alpha=0.5, linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Directional Accuracy (%)', fontsize=11)
    ax.set_title('Directional Accuracy Evolution', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Loss vs Dir Acc (correlation)
    ax = axes[1, 0]
    ax.scatter(train_losses, [acc*100 for acc in train_dir_accs], label='Train', alpha=0.6, s=80, color='blue')
    ax.scatter(val_losses, [acc*100 for acc in val_dir_accs], label='Val', alpha=0.6, s=80, color='red')
    ax.set_xlabel('Loss', fontsize=11)
    ax.set_ylabel('Directional Accuracy (%)', fontsize=11)
    ax.set_title('Loss vs Directional Accuracy', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Summary Statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""
TRAINING SUMMARY - {symbol}
{'='*50}

Total Epochs Trained: {len(epochs)}

FINAL METRICS:
  Train Loss: {train_losses[-1]:.6f}
  Val Loss: {val_losses[-1]:.6f}
  
  Train Dir Acc: {train_dir_accs[-1]*100:.2f}%
  Val Dir Acc: {val_dir_accs[-1]*100:.2f}%

BEST METRICS:
  Best Val Loss: {min(val_losses):.6f} (epoch {np.argmin(val_losses)+1})
  Best Val Dir Acc: {max(val_dir_accs)*100:.2f}% (epoch {np.argmax(val_dir_accs)+1})

IMPROVEMENT:
  Loss Reduction: {((train_losses[0]-train_losses[-1])/train_losses[0]*100):.1f}%
  Dir Acc Gain: {(val_dir_accs[-1]-val_dir_accs[0])*100:.2f}pp

STATUS:
  Random Baseline: 33.33%
  Current Level: {'POOR' if val_dir_accs[-1] < 0.40 else 'FAIR' if val_dir_accs[-1] < 0.50 else 'GOOD' if val_dir_accs[-1] < 0.70 else 'EXCELLENT'}

{'='*50}
    """
    
    ax.text(0.1, 0.95, summary_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(f'results/training_metrics_{symbol}_{model_type}.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[OK] Metrics saved to {output_path}")
    
    # Show in PyCharm
    plt.show()
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"TRAINING SUMMARY - {symbol}")
    print(f"{'='*70}")
    print(f"\nMetrics:")
    print(f"  Final Val Loss: {val_losses[-1]:.6f}")
    print(f"  Final Dir Acc: {val_dir_accs[-1]*100:.2f}%")
    print(f"  Best Dir Acc: {max(val_dir_accs)*100:.2f}% (epoch {np.argmax(val_dir_accs)+1})")
    print(f"\nAnalysis:")
    
    if val_dir_accs[-1] < 0.40:
        print(f"  WARNING: Directional accuracy is LOW ({val_dir_accs[-1]*100:.2f}%)")
        print(f"  Consider: Increase epochs, reduce learning rate, or check features")
    elif val_dir_accs[-1] < 0.50:
        print(f"  FAIR: Model shows some directional ability ({val_dir_accs[-1]*100:.2f}%)")
        print(f"  Recommendation: Continue training or adjust hyperparameters")
    elif val_dir_accs[-1] < 0.70:
        print(f"  GOOD: Model has decent directional accuracy ({val_dir_accs[-1]*100:.2f}%)")
        print(f"  Ready for further optimization")
    else:
        print(f"  EXCELLENT: Strong directional prediction ({val_dir_accs[-1]*100:.2f}%)")
        print(f"  Model is performing very well!")
    
    print(f"\n{'='*70}\n")


def evaluate_model(symbol, model_type='directional'):
    """
    Evaluate trained model on test data
    """
    print(f"\nEvaluating {symbol} model...\n")
    
    model_path = Path(f'models/saved_models/{symbol}_tft_{model_type}_model.pth')
    
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return
    
    try:
        # Fetch data
        print(f"[1/4] Fetching data for {symbol}...")
        fetcher = TFTDataFetcher()
        df = fetcher.fetch_ohlcv_binance(f"{symbol}/USDT", timeframe='1h', limit=3000)
        
        if df is None:
            print("Failed to fetch data")
            return
        
        # Add indicators
        print(f"[2/4] Adding indicators...")
        df = fetcher.add_tft_indicators(df)
        
        # Prepare features
        print(f"[3/4] Preparing features...")
        X, y, scaler = fetcher.prepare_ml_features(df, lookback=60)
        
        if X is None:
            return
        
        # Load model
        print(f"[4/4] Loading model...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = TemporalFusionTransformerV3Optimized(
            input_size=X.shape[2],
            hidden_size=256,
            num_heads=8,
            num_layers=2,
            use_direction_head=True
        ).to(device)
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Test on last 100 samples
        test_size = min(100, len(X))
        X_test = torch.tensor(X[-test_size:], dtype=torch.float32).to(device)
        y_test = y[-test_size:]
        
        with torch.no_grad():
            if model.use_direction_head:
                price_pred, dir_logits = model(X_test, return_direction_logits=True)
                dir_pred = dir_logits.argmax(dim=1).cpu().numpy()
            else:
                price_pred = model(X_test)
                dir_pred = None
            
            price_pred = price_pred.squeeze().cpu().numpy()
        
        # Compute metrics
        mae = np.mean(np.abs(price_pred - y_test))
        mape = np.mean(np.abs((price_pred - y_test) / y_test)) * 100
        rmse = np.sqrt(np.mean((price_pred - y_test) ** 2))
        
        print(f"\n{'='*70}")
        print(f"MODEL EVALUATION - {symbol}")
        print(f"{'='*70}")
        print(f"\nTest Set Size: {test_size} samples")
        print(f"\nPrice Prediction Metrics:")
        print(f"  MAE:  {mae:.6f} USD")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  RMSE: {rmse:.6f} USD")
        
        if dir_pred is not None:
            # Compute directional accuracy
            actual_dir = np.sign(np.diff(y_test, prepend=y_test[0]))
            pred_dir_binary = np.zeros_like(dir_pred)
            pred_dir_binary[dir_pred == 2] = 1
            pred_dir_binary[dir_pred == 0] = -1
            
            dir_acc = np.mean(actual_dir == pred_dir_binary) * 100
            print(f"\nDirectional Metrics:")
            print(f"  Directional Accuracy: {dir_acc:.2f}%")
            print(f"  Random Baseline: 33.33%")
            print(f"  Improvement over Random: {dir_acc - 33.33:.2f}pp")
        
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Visualize Training Results')
    parser.add_argument('--symbol', default='SOL', help='Crypto symbol')
    parser.add_argument('--model', default='directional', choices=['directional', 'standard'])
    parser.add_argument('--eval', action='store_true', help='Evaluate model on test data')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"Training Results Visualization - {args.symbol}")
    print(f"{'='*70}\n")
    
    # Plot metrics
    plot_training_metrics(args.symbol, args.model)
    
    # Evaluate if requested
    if args.eval:
        evaluate_model(args.symbol, args.model)


if __name__ == '__main__':
    main()
