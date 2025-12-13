#!/usr/bin/env python3
"""
🎯 Real-Time Training Monitor

Live dashboard for monitoring TFT V3 training progress
Shows:
  - Training/Validation loss curves
  - Learning rate changes
  - Model performance metrics
  - ETA and progress
  - Early stopping status
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
import argparse

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("Error: matplotlib and numpy required")
    print("Install with: pip install matplotlib numpy")
    exit(1)


class TrainingMonitor:
    """Monitor training progress in real-time"""
    
    def __init__(self, log_file='logs/training_tft_v3.log', refresh_interval=10):
        self.log_file = log_file
        self.refresh_interval = refresh_interval
        self.epoch_data = {'epoch': [], 'train_loss': [], 'val_loss': [], 'lr': []}
        self.last_position = 0
    
    def parse_log(self):
        """Parse training log file"""
        if not os.path.exists(self.log_file):
            return None
        
        try:
            with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(self.last_position)
                lines = f.readlines()
                self.last_position = f.tell()
            
            new_data = False
            for line in lines:
                # Parse epoch logs like: "Epoch  10/200 | Train Loss: 0.002345 | Val Loss: 0.002567 | LR: 9.97e-05"
                if 'Epoch' in line and 'Train Loss' in line:
                    try:
                        # Extract values
                        parts = line.split('|')
                        epoch_str = parts[0].split()[-2].split('/')[0]  # "10" from "10/200"
                        train_loss = float(parts[1].split(':')[1].strip())
                        val_loss = float(parts[2].split(':')[1].strip())
                        lr = float(parts[3].split(':')[1].strip())
                        
                        epoch = int(epoch_str)
                        
                        if not self.epoch_data['epoch'] or epoch > self.epoch_data['epoch'][-1]:
                            self.epoch_data['epoch'].append(epoch)
                            self.epoch_data['train_loss'].append(train_loss)
                            self.epoch_data['val_loss'].append(val_loss)
                            self.epoch_data['lr'].append(lr)
                            new_data = True
                    except (ValueError, IndexError):
                        pass
            
            return new_data
        except Exception as e:
            print(f"Error parsing log: {e}")
            return None
    
    def get_status(self):
        """Get current training status"""
        if not self.epoch_data['epoch']:
            return None
        
        status = {
            'current_epoch': self.epoch_data['epoch'][-1],
            'train_loss': self.epoch_data['train_loss'][-1],
            'val_loss': self.epoch_data['val_loss'][-1],
            'lr': self.epoch_data['lr'][-1],
            'best_val_loss': min(self.epoch_data['val_loss']),
            'epochs_run': len(self.epoch_data['epoch']),
        }
        
        # Calculate improvement
        if len(self.epoch_data['val_loss']) > 1:
            initial_loss = self.epoch_data['val_loss'][0]
            current_loss = self.epoch_data['val_loss'][-1]
            improvement = ((initial_loss - current_loss) / initial_loss) * 100
            status['improvement_pct'] = improvement
        
        return status
    
    def plot_progress(self, output_file='training_progress.png'):
        """Generate training progress plots"""
        if not self.epoch_data['epoch']:
            print("No data to plot yet")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('TFT V3 Training Monitor', fontsize=16, fontweight='bold')
        
        epochs = self.epoch_data['epoch']
        train_losses = self.epoch_data['train_loss']
        val_losses = self.epoch_data['val_loss']
        lrs = self.epoch_data['lr']
        
        # Plot 1: Loss curves
        ax = axes[0, 0]
        ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2, marker='o', markersize=4)
        ax.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2, marker='s', markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training & Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Loss improvement
        ax = axes[0, 1]
        val_improvement = [(val_losses[0] - loss) / val_losses[0] * 100 for loss in val_losses]
        ax.fill_between(epochs, 0, val_improvement, alpha=0.3, color='green')
        ax.plot(epochs, val_improvement, 'g-', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Improvement (%)')
        ax.set_title('Validation Loss Improvement')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Learning rate
        ax = axes[1, 0]
        ax.semilogy(epochs, lrs, 'purple', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Statistics
        ax = axes[1, 1]
        ax.axis('off')
        
        status = self.get_status()
        if status:
            stats_text = f"""
 TRAINING STATISTICS
 ═══════════════════════════════════
 
 Current Epoch: {status['current_epoch']}
 Epochs Run: {status['epochs_run']}
 
 Train Loss: {status['train_loss']:.6f}
 Val Loss: {status['val_loss']:.6f}
 Best Val Loss: {status['best_val_loss']:.6f}
 
 Learning Rate: {status['lr']:.2e}
 
 Improvement: {status['improvement_pct']:.1f}%
 
 """
            
            ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
                   fontsize=11, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
        return output_file
    
    def print_status(self):
        """Print current status to console"""
        status = self.get_status()
        
        if not status:
            print("Waiting for training to start...")
            return
        
        print("\n" + "="*70)
        print(f"TFT V3 Training Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        print(f"\nEpoch: {status['current_epoch']} (Total: {status['epochs_run']} epochs)")
        print(f"\nLoss:")
        print(f"  Train: {status['train_loss']:.6f}")
        print(f"  Val:   {status['val_loss']:.6f}")
        print(f"  Best:  {status['best_val_loss']:.6f}")
        print(f"\nLearning Rate: {status['lr']:.2e}")
        print(f"Improvement: {status['improvement_pct']:.1f}%")
        print("="*70 + "\n")
    
    def monitor_live(self, duration_minutes=0):
        """Monitor training in real-time"""
        print(f"Starting live monitor (refresh every {self.refresh_interval}s)")
        print(f"Press Ctrl+C to stop\n")
        
        start_time = time.time()
        duration_seconds = duration_minutes * 60 if duration_minutes > 0 else float('inf')
        
        iteration = 0
        while True:
            iteration += 1
            
            # Check elapsed time
            elapsed = time.time() - start_time
            if elapsed > duration_seconds:
                print(f"\nMonitoring duration ({duration_minutes}m) reached")
                break
            
            # Parse log and update
            new_data = self.parse_log()
            
            # Print status
            self.print_status()
            
            # Generate plot every 10 iterations
            if iteration % 10 == 0:
                self.plot_progress()
            
            # Wait before next update
            try:
                time.sleep(self.refresh_interval)
            except KeyboardInterrupt:
                print("\nMonitoring stopped by user")
                break


def main():
    parser = argparse.ArgumentParser(description='TFT V3 Training Monitor')
    parser.add_argument('--log', default='logs/training_tft_v3.log', help='Log file path')
    parser.add_argument('--interval', type=int, default=10, help='Refresh interval (seconds)')
    parser.add_argument('--duration', type=int, default=0, help='Monitor duration (minutes, 0=infinite)')
    parser.add_argument('--plot-only', action='store_true', help='Generate plot and exit')
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor(args.log, args.interval)
    
    if args.plot_only:
        # Just parse and plot once
        monitor.parse_log()
        monitor.plot_progress()
    else:
        # Live monitoring
        monitor.monitor_live(args.duration)
        # Final plot
        monitor.plot_progress()


if __name__ == '__main__':
    main()
