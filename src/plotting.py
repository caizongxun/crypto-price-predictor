import matplotlib.pyplot as plt
import io
import numpy as np

def generate_prediction_chart(symbol: str, history: np.ndarray, prediction: list, timeframe_label: str = "1h") -> io.BytesIO:
    """
    Generate a chart showing historical prices and predicted future path.
    """
    try:
        plt.style.use('dark_background')
        fig, ax = plt.figure(figsize=(10, 5)), plt.gca()
        
        # History (take last 30 points for clarity)
        history_subset = history[-30:]
        x_hist = range(len(history_subset))
        ax.plot(x_hist, history_subset, label='History', color='#00ffcc', linewidth=2)
        
        # Prediction
        # Start prediction line from the last history point to make it continuous
        last_hist_val = history_subset[-1]
        last_hist_idx = x_hist[-1]
        
        pred_values = [last_hist_val] + prediction
        pred_x = range(last_hist_idx, last_hist_idx + len(pred_values))
        
        ax.plot(pred_x, pred_values, label='AI Prediction', color='#ff00cc', linestyle='--', linewidth=2, marker='o')
        
        # Formatting
        ax.set_title(f"{symbol} Price Prediction ({timeframe_label})", fontsize=14, pad=15)
        ax.grid(True, alpha=0.2)
        ax.legend(loc='upper left')
        
        # Remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        plt.close(fig)
        
        return buf
    except Exception as e:
        print(f"Error generating chart: {e}")
        return None
