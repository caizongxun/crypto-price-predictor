import discord
from discord.ext import commands, tasks
import logging
from datetime import datetime
from typing import Optional, List
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class TrainingNotificationCog(commands.Cog):
    """Discord Cog for training notifications"""
    
    def __init__(self, bot):
        self.bot = bot
        self.channel_id = int(os.getenv('DISCORD_CHANNEL_ID', '0'))
        self.role_id = int(os.getenv('DISCORD_ALERT_ROLE_ID', '0'))
    
    @commands.Cog.listener()
    async def on_ready(self):
        """Called when bot is ready"""
        logger.info(f'{self.bot.user} has connected to Discord!')
        print(f'✅ Discord Bot is online: {self.bot.user}')
    
    async def send_training_notification(
        self,
        symbol: str,
        epochs: int,
        train_loss: float,
        val_loss: float,
        training_time: float,
        success: bool = True,
        error_msg: Optional[str] = None
    ) -> bool:
        """Send training completion notification"""
        try:
            channel = self.bot.get_channel(self.channel_id)
            if not channel:
                logger.error(f"Channel {self.channel_id} not found")
                return False
            
            # Determine emoji and color
            if success:
                emoji = "✅"
                color = discord.Color.green()
                status = "SUCCESS"
                mention = f"<@&{self.role_id}>" if self.role_id else ""
            else:
                emoji = "❌"
                color = discord.Color.red()
                status = "FAILED"
                mention = f"<@&{self.role_id}>" if self.role_id else ""
            
            # Create embed
            embed = discord.Embed(
                title=f"{emoji} {symbol} Training {status}",
                description=f"Model training completed for {symbol}",
                color=color,
                timestamp=datetime.now()
            )
            
            # Add fields
            embed.add_field(name="Symbol", value=symbol, inline=True)
            embed.add_field(name="Status", value=status, inline=True)
            embed.add_field(name="Epochs", value=str(epochs), inline=True)
            embed.add_field(name="Train Loss", value=f"{train_loss:.6f}", inline=True)
            embed.add_field(name="Val Loss", value=f"{val_loss:.6f}", inline=True)
            embed.add_field(name="Training Time", value=f"{training_time:.1f}s", inline=True)
            
            if error_msg:
                embed.add_field(name="Error", value=f"```{error_msg[:200]}```", inline=False)
            
            # Expected accuracy
            if val_loss < 0.025:
                accuracy = "🎯 ~90%+"
            elif val_loss < 0.035:
                accuracy = "⚖️ ~85-90%"
            else:
                accuracy = "📊 <85%"
            embed.add_field(name="Expected Accuracy", value=accuracy, inline=True)
            
            embed.set_footer(text=f"Crypto Price Predictor | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Send message
            message_content = mention if mention else ""
            await channel.send(content=message_content, embed=embed)
            
            logger.info(f"Training notification sent for {symbol}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to send training notification: {e}")
            return False
    
    async def send_batch_training_notification(
        self,
        symbols: List[str],
        successful_count: int,
        failed_count: int,
        total_time: float,
        avg_val_loss: float,
        detailed_results: Optional[List[dict]] = None
    ) -> bool:
        """Send batch training completion notification"""
        try:
            channel = self.bot.get_channel(self.channel_id)
            if not channel:
                logger.error(f"Channel {self.channel_id} not found")
                return False
            
            # Determine status
            if failed_count == 0:
                emoji = "🎉"
                color = discord.Color.green()
                status = "COMPLETE"
                mention = f"<@&{self.role_id}>" if self.role_id else ""
            else:
                emoji = "⚠️"
                color = discord.Color.orange()
                status = "PARTIAL"
                mention = f"<@&{self.role_id}>" if self.role_id else ""
            
            # Create embed
            embed = discord.Embed(
                title=f"{emoji} Batch Training {status}",
                description=f"Completed training for {len(symbols)} cryptocurrencies",
                color=color,
                timestamp=datetime.now()
            )
            
            # Add summary fields
            embed.add_field(name="Total Symbols", value=str(len(symbols)), inline=True)
            embed.add_field(name="Successful", value=f"✅ {successful_count}", inline=True)
            embed.add_field(name="Failed", value=f"❌ {failed_count}", inline=True)
            embed.add_field(name="Success Rate", value=f"{(successful_count/len(symbols)*100):.1f}%", inline=True)
            embed.add_field(name="Avg Val Loss", value=f"{avg_val_loss:.6f}", inline=True)
            embed.add_field(name="Total Time", value=f"{total_time/60:.1f} min", inline=True)
            
            # Add symbols list
            symbols_str = ", ".join(symbols)
            if len(symbols_str) > 1024:
                symbols_str = symbols_str[:1021] + "..."
            embed.add_field(name="Symbols", value=symbols_str, inline=False)
            
            # Add detailed results if provided
            if detailed_results:
                results_text = ""
                for result in detailed_results[:10]:  # Show top 10
                    symbol = result.get('symbol', 'N/A')
                    status_emoji = "✅" if result.get('success') else "❌"
                    val_loss = result.get('val_loss', 0)
                    results_text += f"{status_emoji} {symbol}: {val_loss:.6f}\n"
                
                if len(detailed_results) > 10:
                    results_text += f"... and {len(detailed_results) - 10} more"
                
                embed.add_field(name="Results Summary", value=f"```{results_text}```", inline=False)
            
            embed.set_footer(text=f"Crypto Price Predictor | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Send message
            message_content = mention if mention else ""
            await channel.send(content=message_content, embed=embed)
            
            logger.info("Batch training notification sent")
            return True
        
        except Exception as e:
            logger.error(f"Failed to send batch training notification: {e}")
            return False
    
    async def send_status_update(
        self,
        title: str,
        description: str,
        fields: dict,
        color: discord.Color = discord.Color.blue()
    ) -> bool:
        """Send custom status update"""
        try:
            channel = self.bot.get_channel(self.channel_id)
            if not channel:
                logger.error(f"Channel {self.channel_id} not found")
                return False
            
            embed = discord.Embed(
                title=title,
                description=description,
                color=color,
                timestamp=datetime.now()
            )
            
            for field_name, field_value in fields.items():
                embed.add_field(name=field_name, value=str(field_value), inline=True)
            
            embed.set_footer(text=f"Crypto Price Predictor | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            await channel.send(embed=embed)
            logger.info(f"Status update sent: {title}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to send status update: {e}")
            return False


class DiscordBotHandler:
    """Main Discord Bot Handler"""
    
    def __init__(self):
        self.token = os.getenv('DISCORD_BOT_TOKEN')
        self.channel_id = os.getenv('DISCORD_CHANNEL_ID')
        
        # Create bot with intents
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        
        self.bot = commands.Bot(command_prefix='!', intents=intents)
        
        # Add cogs
        asyncio.get_event_loop().run_until_complete(self._setup_cogs())
        
        self.training_cog: Optional[TrainingNotificationCog] = None
    
    async def _setup_cogs(self):
        """Setup bot cogs"""
        await self.bot.add_cog(TrainingNotificationCog(self.bot))
    
    def start(self):
        """Start the bot"""
        if not self.token:
            logger.error("DISCORD_BOT_TOKEN not configured")
            return False
        
        try:
            self.bot.run(self.token)
            return True
        except Exception as e:
            logger.error(f"Failed to start Discord bot: {e}")
            return False
    
    async def send_training_notification(
        self,
        symbol: str,
        epochs: int,
        train_loss: float,
        val_loss: float,
        training_time: float,
        success: bool = True,
        error_msg: Optional[str] = None
    ) -> bool:
        """Send training notification via bot"""
        cog = self.bot.get_cog('TrainingNotificationCog')
        if cog:
            return await cog.send_training_notification(
                symbol, epochs, train_loss, val_loss, training_time, success, error_msg
            )
        return False
    
    async def send_batch_training_notification(
        self,
        symbols: List[str],
        successful_count: int,
        failed_count: int,
        total_time: float,
        avg_val_loss: float,
        detailed_results: Optional[List[dict]] = None
    ) -> bool:
        """Send batch training notification via bot"""
        cog = self.bot.get_cog('TrainingNotificationCog')
        if cog:
            return await cog.send_batch_training_notification(
                symbols, successful_count, failed_count, total_time, avg_val_loss, detailed_results
            )
        return False


import asyncio
