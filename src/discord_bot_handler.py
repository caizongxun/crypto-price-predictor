import discord
from discord.ext import commands, tasks
import logging
from datetime import datetime
from typing import Optional, List
import os
from dotenv import load_dotenv
import asyncio
from queue import Queue
import threading
import sys

load_dotenv()
logger = logging.getLogger(__name__)


class TrainingNotificationCog(commands.Cog):
    """Discord Cog for trading signal and training notifications"""
    
    def __init__(self, bot):
        self.bot = bot
        self.channel_id = int(os.getenv('DISCORD_CHANNEL_ID', '0'))
        self.role_id = int(os.getenv('DISCORD_ALERT_ROLE_ID', '0'))
        self.embed_queue = Queue()
        
        if self.channel_id > 0:
            self.process_queue.start()
            logger.info(f"✅ Queue processor started for channel {self.channel_id}")
        else:
            logger.warning("⚠️ DISCORD_CHANNEL_ID not set - notifications disabled")
    
    @tasks.loop(seconds=2)
    async def process_queue(self):
        """Process embed queue and send messages"""
        try:
            while not self.embed_queue.empty():
                try:
                    embed = self.embed_queue.get_nowait()
                    channel = self.bot.get_channel(self.channel_id)
                    if channel:
                        try:
                            await channel.send(embed=embed)
                            logger.debug(f"✅ Sent embed to {channel.name}")
                        except discord.errors.Forbidden:
                            logger.error(f"❌ No permission to send message in channel {self.channel_id}")
                        except Exception as e:
                            logger.error(f"❌ Failed to send embed: {e}")
                    else:
                        logger.error(f"❌ Channel {self.channel_id} not found")
                except Exception as e:
                    logger.error(f"Error getting from queue: {e}")
                    break
        except Exception as e:
            logger.error(f"❌ Error processing embed queue: {e}")
    
    @process_queue.before_loop
    async def before_process_queue(self):
        """Wait for bot to be ready before processing queue"""
        await self.bot.wait_until_ready()
        logger.info("✅ Queue processor ready")
    
    @commands.Cog.listener()
    async def on_ready(self):
        """Called when bot is ready"""
        logger.info(f'✅ Discord Bot is online: {self.bot.user}')
        print(f'✅ Discord Bot is online: {self.bot.user}')
    
    @commands.command(name='recommendation', help='Get latest trading recommendation')
    async def recommendation_command(self, ctx):
        """Get latest trading recommendations"""
        try:
            embed = discord.Embed(
                title="📊 Latest Trading Signals",
                description="Loading latest signals...",
                color=discord.Color.blue()
            )
            await ctx.send(embed=embed)
            logger.info(f"✅ Recommendation command executed by {ctx.author}")
        except Exception as e:
            logger.error(f"Error in recommendation command: {e}")
            await ctx.send(f"❌ Error: {e}")
    
    @commands.command(name='status', help='Get bot status')
    async def status_command(self, ctx):
        """Get bot status"""
        try:
            embed = discord.Embed(
                title="🤖 Bot Status",
                color=discord.Color.green()
            )
            embed.add_field(name="Status", value="✅ Online", inline=True)
            embed.add_field(name="Latency", value=f"{self.bot.latency*1000:.0f}ms", inline=True)
            await ctx.send(embed=embed)
            logger.info(f"✅ Status command executed by {ctx.author}")
        except Exception as e:
            logger.error(f"Error in status command: {e}")
            await ctx.send(f"❌ Error: {e}")
    
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
            
            logger.info(f"✅ Training notification sent for {symbol}")
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
            
            logger.info("✅ Batch training notification sent")
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
            logger.info(f"✅ Status update sent: {title}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to send status update: {e}")
            return False


class DiscordBotHandler:
    """Main Discord Bot Handler with proper async/threading support"""
    
    def __init__(self):
        self.token = os.getenv('DISCORD_BOT_TOKEN')
        self.channel_id = os.getenv('DISCORD_CHANNEL_ID')
        
        if not self.token:
            logger.error("❌ DISCORD_BOT_TOKEN not configured")
            self.bot = None
            self.training_cog = None
            return
        
        # Create bot with intents
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        
        self.bot = commands.Bot(command_prefix='!', intents=intents)
        self.training_cog = None
        self.loop = None
        self.thread = None
    
    def _run_bot(self):
        """Run bot in separate thread with its own event loop"""
        try:
            # Create new event loop for this thread
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            # Add cog to bot
            async def setup():
                cog = TrainingNotificationCog(self.bot)
                await self.bot.add_cog(cog)
                self.training_cog = cog
                logger.info("✅ TrainingNotificationCog loaded")
            
            # Run setup
            self.loop.run_until_complete(setup())
            
            # Start bot
            self.loop.run_until_complete(self.bot.start(self.token))
        
        except Exception as e:
            logger.error(f"❌ Discord bot error: {e}", exc_info=True)
        finally:
            try:
                self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            except:
                pass
    
    def start(self):
        """Start the bot in background thread"""
        if not self.token:
            logger.error("❌ DISCORD_BOT_TOKEN not set")
            return
        
        try:
            # Start bot in daemon thread
            self.thread = threading.Thread(target=self._run_bot, daemon=True)
            self.thread.start()
            logger.info("✅ Discord bot thread started")
        except Exception as e:
            logger.error(f"❌ Failed to start Discord bot thread: {e}")
    
    def queue_embed(self, embed: discord.Embed):
        """Queue an embed to be sent (thread-safe)"""
        if self.training_cog:
            try:
                self.training_cog.embed_queue.put(embed)
                logger.debug(f"✅ Embed queued (queue size: {self.training_cog.embed_queue.qsize()})")
            except Exception as e:
                logger.error(f"❌ Error queueing embed: {e}")
        else:
            logger.warning("⚠️ TrainingNotificationCog not available yet, embed discarded")
    
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
        if self.training_cog:
            return await self.training_cog.send_training_notification(
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
        if self.training_cog:
            return await self.training_cog.send_batch_training_notification(
                symbols, successful_count, failed_count, total_time, avg_val_loss, detailed_results
            )
        return False
