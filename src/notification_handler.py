import os
import smtplib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class NotificationHandler:
    """Handle notifications via Telegram, Email, Discord"""
    
    def __init__(self):
        # Telegram
        self.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        # Email
        self.email_sender = os.getenv('EMAIL_SENDER')
        self.email_password = os.getenv('EMAIL_PASSWORD')
        self.email_recipient = os.getenv('EMAIL_RECIPIENT')
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        
        # Discord
        self.discord_webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
    
    def send_telegram_notification(self, message: str) -> bool:
        """Send notification via Telegram"""
        if not self.telegram_bot_token or not self.telegram_chat_id:
            logger.warning("Telegram credentials not configured")
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            payload = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                logger.info("Telegram notification sent successfully")
                return True
            else:
                logger.error(f"Telegram error: {response.text}")
                return False
        
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")
            return False
    
    def send_email_notification(self, subject: str, message: str) -> bool:
        """Send notification via Email"""
        if not self.email_sender or not self.email_password or not self.email_recipient:
            logger.warning("Email credentials not configured")
            return False
        
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.email_sender
            msg['To'] = self.email_recipient
            
            # Attach HTML version
            html_part = MIMEText(message, 'html')
            msg.attach(html_part)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_sender, self.email_password)
                server.send_message(msg)
            
            logger.info("Email notification sent successfully")
            return True
        
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False
    
    def send_discord_notification(self, embed_data: Dict) -> bool:
        """Send notification via Discord Webhook"""
        if not self.discord_webhook_url:
            logger.warning("Discord webhook URL not configured")
            return False
        
        try:
            payload = {'embeds': [embed_data]}
            response = requests.post(self.discord_webhook_url, json=payload, timeout=10)
            
            if response.status_code in [200, 204]:
                logger.info("Discord notification sent successfully")
                return True
            else:
                logger.error(f"Discord error: {response.text}")
                return False
        
        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")
            return False
    
    def send_training_complete_notification(
        self,
        symbol: str,
        epochs: int,
        train_loss: float,
        val_loss: float,
        training_time: float,
        success: bool = True
    ) -> None:
        """Send notification when training completes for a symbol"""
        
        status = "SUCCESS" if success else "FAILED"
        status_emoji = "✅" if success else "❌"
        
        # Telegram message
        telegram_message = f"""
<b>{status_emoji} {symbol} Training {status}</b>

Epochs: {epochs}
Train Loss: {train_loss:.6f}
Val Loss: {val_loss:.6f}
Time: {training_time:.1f} seconds
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        # Email message
        email_subject = f"[{status}] Crypto Model Training - {symbol}"
        email_message = f"""
<html>
<body>
<h2>{status_emoji} {symbol} Training {status}</h2>
<table border="1" cellpadding="10">
<tr><td><b>Epochs</b></td><td>{epochs}</td></tr>
<tr><td><b>Train Loss</b></td><td>{train_loss:.6f}</td></tr>
<tr><td><b>Val Loss</b></td><td>{val_loss:.6f}</td></tr>
<tr><td><b>Training Time</b></td><td>{training_time:.1f} seconds</td></tr>
<tr><td><b>Timestamp</b></td><td>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
</table>
</body>
</html>
        """
        
        # Discord embed
        discord_embed = {
            'title': f'{symbol} Training {status}',
            'description': f'Model training completed for {symbol}',
            'color': 3066993 if success else 15158332,  # Green or Red
            'fields': [
                {'name': 'Epochs', 'value': str(epochs), 'inline': True},
                {'name': 'Train Loss', 'value': f'{train_loss:.6f}', 'inline': True},
                {'name': 'Val Loss', 'value': f'{val_loss:.6f}', 'inline': True},
                {'name': 'Time', 'value': f'{training_time:.1f}s', 'inline': True},
                {'name': 'Status', 'value': status, 'inline': True},
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        # Send notifications
        self.send_telegram_notification(telegram_message)
        self.send_email_notification(email_subject, email_message)
        self.send_discord_notification(discord_embed)
    
    def send_batch_training_complete_notification(
        self,
        symbols: List[str],
        successful_count: int,
        failed_count: int,
        total_time: float,
        avg_val_loss: float
    ) -> None:
        """Send notification when batch training completes"""
        
        status = "SUCCESS" if failed_count == 0 else "PARTIAL"
        status_emoji = "✅" if failed_count == 0 else "⚠️"
        
        # Telegram message
        telegram_message = f"""
<b>{status_emoji} Batch Training {status}</b>

Total Symbols: {len(symbols)}
Successful: {successful_count}
Failed: {failed_count}
Average Val Loss: {avg_val_loss:.6f}
Total Time: {total_time/60:.1f} minutes

Symbols: {', '.join(symbols)}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        # Email message
        email_subject = f"[{status}] Batch Training Complete - {len(symbols)} Cryptocurrencies"
        email_message = f"""
<html>
<body>
<h2>{status_emoji} Batch Training {status}</h2>
<table border="1" cellpadding="10">
<tr><td><b>Total Symbols</b></td><td>{len(symbols)}</td></tr>
<tr><td><b>Successful</b></td><td>{successful_count}</td></tr>
<tr><td><b>Failed</b></td><td>{failed_count}</td></tr>
<tr><td><b>Average Val Loss</b></td><td>{avg_val_loss:.6f}</td></tr>
<tr><td><b>Total Time</b></td><td>{total_time/60:.1f} minutes</td></tr>
<tr><td><b>Symbols</b></td><td>{', '.join(symbols)}</td></tr>
<tr><td><b>Timestamp</b></td><td>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
</table>
</body>
</html>
        """
        
        # Discord embed
        discord_embed = {
            'title': f'Batch Training {status}',
            'description': f'Completed training for {len(symbols)} cryptocurrencies',
            'color': 3066993 if failed_count == 0 else 15105570,  # Green or Orange
            'fields': [
                {'name': 'Total', 'value': str(len(symbols)), 'inline': True},
                {'name': 'Successful', 'value': str(successful_count), 'inline': True},
                {'name': 'Failed', 'value': str(failed_count), 'inline': True},
                {'name': 'Avg Val Loss', 'value': f'{avg_val_loss:.6f}', 'inline': True},
                {'name': 'Time', 'value': f'{total_time/60:.1f} min', 'inline': True},
                {'name': 'Symbols', 'value': ', '.join(symbols), 'inline': False},
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        # Send notifications
        self.send_telegram_notification(telegram_message)
        self.send_email_notification(email_subject, email_message)
        self.send_discord_notification(discord_embed)
