"""Discord bot for sending cryptocurrency trading signals and notifications."""

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime

import discord
from discord.ext import commands, tasks
from dotenv import load_dotenv
import os

load_dotenv()
logger = logging.getLogger(__name__)


class DiscordBot(commands.Cog):
    """Discord bot for trading signal notifications."""
    
    def __init__(self, bot: commands.Bot, config: Dict = None):
        """Initialize Discord bot.
        
        Args:
            bot: Discord bot instance
            config: Configuration dictionary
        """
        self.bot = bot
        self.config = config or {}
        self.channel_id = int(os.getenv('DISCORD_CHANNEL_ID', 0))
        self.role_id = os.getenv('DISCORD_ALERT_ROLE_ID')
        self.predictor = None
        
    @commands.command(name='predict', help='Get price prediction for a cryptocurrency')
    async def predict_command(self, ctx, symbol: str):
        """Get price prediction.
        
        Args:
            ctx: Command context
            symbol: Cryptocurrency symbol (e.g., BTC)
        """
        try:
            if self.predictor is None:
                await ctx.send("Predictor not initialized yet.")
                return
            
            # This would be implemented based on your data fetcher
            embed = discord.Embed(
                title=f"{symbol} Price Prediction",
                color=discord.Color.blue(),
                timestamp=datetime.now()
            )
            
            embed.add_field(name="Status", value="Fetching prediction...", inline=False)
            await ctx.send(embed=embed)
        except Exception as e:
            logger.error(f"Error in predict command: {e}")
            await ctx.send(f"Error: {str(e)}")
    
    @commands.command(name='signal', help='Get trading signal for a cryptocurrency')
    async def signal_command(self, ctx, symbol: str):
        """Get trading signal.
        
        Args:
            ctx: Command context
            symbol: Cryptocurrency symbol
        """
        try:
            if self.predictor is None:
                await ctx.send("Predictor not initialized yet.")
                return
            
            embed = discord.Embed(
                title=f"{symbol} Trading Signal",
                color=discord.Color.green(),
                timestamp=datetime.now()
            )
            
            embed.add_field(name="Status", value="Generating signal...", inline=False)
            await ctx.send(embed=embed)
        except Exception as e:
            logger.error(f"Error in signal command: {e}")
            await ctx.send(f"Error: {str(e)}")
    
    async def send_signal_notification(self, signal: Dict):
        """Send trading signal as Discord message.
        
        Args:
            signal: Trading signal dictionary
        """
        try:
            channel = self.bot.get_channel(self.channel_id)
            if not channel:
                logger.error(f"Channel {self.channel_id} not found")
                return
            
            symbol = signal.get('symbol', 'UNKNOWN')
            recommendation = signal.get('recommendation', 'HOLD')
            current_price = signal.get('current_price', 0)
            predicted_price = signal.get('predicted_price', 0)
            predicted_change = signal.get('predicted_change', 0)
            confidence = signal.get('confidence', 0)
            
            # Determine color based on recommendation
            color_map = {
                'BUY': discord.Color.green(),
                'BUY_WEAK': discord.Color.light_gray(),
                'SELL': discord.Color.red(),
                'SELL_WEAK': discord.Color.dark_red(),
                'HOLD': discord.Color.orange(),
                'WAIT': discord.Color.greyple()
            }
            
            color = color_map.get(recommendation, discord.Color.blue())
            
            embed = discord.Embed(
                title=f"🚀 {symbol} Trading Signal",
                description=f"**Recommendation: {recommendation}**",
                color=color,
                timestamp=datetime.now()
            )
            
            embed.add_field(name="Current Price", value=f"${current_price:.2f}", inline=True)
            embed.add_field(name="Predicted Price", value=f"${predicted_price:.2f}", inline=True)
            embed.add_field(name="Expected Change", value=f"{predicted_change:+.2f}%", inline=True)
            
            embed.add_field(name="Confidence", value=f"{confidence:.1f}%", inline=True)
            embed.add_field(name="Support Level", value=f"${signal.get('support_level', 0):.2f}", inline=True)
            embed.add_field(name="Resistance Level", value=f"${signal.get('resistance_level', 0):.2f}", inline=True)
            
            # Entry recommendations
            entry_min = signal.get('entry_min', 0)
            entry_max = signal.get('entry_max', 0)
            embed.add_field(
                name="Entry Zone",
                value=f"${entry_min:.2f} - ${entry_max:.2f}",
                inline=False
            )
            
            # Take profit levels
            tp_levels = signal.get('take_profit', [])
            tp_text = "\n".join([f"TP {i+1}: ${tp:.2f}" for i, tp in enumerate(tp_levels)])
            embed.add_field(name="Take Profit Levels", value=tp_text, inline=False)
            
            # Stop loss
            stop_loss = signal.get('stop_loss', 0)
            embed.add_field(name="Stop Loss", value=f"${stop_loss:.2f}", inline=False)
            
            # Technical indicators
            rsi = signal.get('rsi', 50)
            macd = signal.get('macd', 0)
            trend = signal.get('trend', 'NEUTRAL')
            
            embed.add_field(name="RSI", value=f"{rsi:.1f}", inline=True)
            embed.add_field(name="MACD", value=f"{macd:+.4f}", inline=True)
            embed.add_field(name="Trend", value=trend, inline=True)
            
            # Add disclaimer
            embed.set_footer(text="⚠️ Not financial advice. Do your own research (DYOR). Cryptocurrency is risky.")
            
            # Mention role if configured
            mention_text = ""
            if self.role_id:
                mention_text = f"<@&{self.role_id}> "
            
            await channel.send(mention_text, embed=embed)
            logger.info(f"Sent signal notification for {symbol}")
        except Exception as e:
            logger.error(f"Failed to send signal notification: {e}")
    
    async def send_price_alert(self, symbol: str, current_price: float,
                               alert_type: str = "price_target", data: Dict = None):
        """Send price alert.
        
        Args:
            symbol: Cryptocurrency symbol
            current_price: Current price
            alert_type: Type of alert
            data: Additional data
        """
        try:
            channel = self.bot.get_channel(self.channel_id)
            if not channel:
                return
            
            embed = discord.Embed(
                title=f"📊 {symbol} Price Alert",
                color=discord.Color.blue(),
                timestamp=datetime.now()
            )
            
            embed.add_field(name="Current Price", value=f"${current_price:.2f}", inline=False)
            
            if data:
                for key, value in data.items():
                    embed.add_field(name=key, value=value, inline=True)
            
            await channel.send(embed=embed)
            logger.info(f"Sent price alert for {symbol}")
        except Exception as e:
            logger.error(f"Failed to send price alert: {e}")
    
    async def send_error_notification(self, error_message: str):
        """Send error notification.
        
        Args:
            error_message: Error message
        """
        try:
            channel = self.bot.get_channel(self.channel_id)
            if not channel:
                return
            
            embed = discord.Embed(
                title="❌ Error Notification",
                description=error_message,
                color=discord.Color.red(),
                timestamp=datetime.now()
            )
            
            await channel.send(embed=embed)
            logger.error(f"Sent error notification: {error_message}")
        except Exception as e:
            logger.error(f"Failed to send error notification: {e}")
    
    async def send_summary(self, crypto_data: List[Dict]):
        """Send daily/hourly summary.
        
        Args:
            crypto_data: List of cryptocurrency data
        """
        try:
            channel = self.bot.get_channel(self.channel_id)
            if not channel:
                return
            
            embed = discord.Embed(
                title="📈 Cryptocurrency Summary",
                color=discord.Color.purple(),
                timestamp=datetime.now()
            )
            
            for crypto in crypto_data:
                symbol = crypto.get('symbol', 'UNKNOWN')
                price = crypto.get('price', 0)
                change = crypto.get('change_percent', 0)
                
                change_emoji = "🟢" if change > 0 else "🔴"
                embed.add_field(
                    name=f"{change_emoji} {symbol}",
                    value=f"${price:.2f} ({change:+.2f}%)",
                    inline=True
                )
            
            await channel.send(embed=embed)
            logger.info("Sent summary notification")
        except Exception as e:
            logger.error(f"Failed to send summary: {e}")


async def setup_discord_bot(config: Dict = None) -> commands.Bot:
    """Setup and initialize Discord bot.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized bot instance
    """
    try:
        intents = discord.Intents.default()
        intents.message_content = True
        
        bot = commands.Bot(command_prefix='/', intents=intents)
        
        @bot.event
        async def on_ready():
            logger.info(f"Discord bot logged in as {bot.user}")
        
        # Add cog
        await bot.add_cog(DiscordBot(bot, config))
        
        return bot
    except Exception as e:
        logger.error(f"Failed to setup Discord bot: {e}")
        return None


async def start_discord_bot(bot: commands.Bot, token: str):
    """Start the Discord bot.
    
    Args:
        bot: Bot instance
        token: Discord bot token
    """
    try:
        await bot.start(token)
    except Exception as e:
        logger.error(f"Failed to start Discord bot: {e}")
