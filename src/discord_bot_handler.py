import discord
from discord.ext import commands, tasks
import logging
from datetime import datetime
from typing import Optional, List, Dict
import os
from dotenv import load_dotenv
import asyncio
from queue import Queue
import threading
import sys

load_dotenv()
logger = logging.getLogger(__name__)

# Global signal storage
GLOBAL_SIGNALS: Dict = {}


class TrainingNotificationCog(commands.Cog):
    """Discord Cog for trading signal and training notifications"""
    
    def __init__(self, bot):
        self.bot = bot
        self.channel_id = int(os.getenv('DISCORD_CHANNEL_ID', '0'))
        self.role_id = int(os.getenv('DISCORD_ALERT_ROLE_ID', '0'))
        # Updated queue to hold (embed, file) tuples
        self.message_queue = Queue()
        
        if self.channel_id > 0:
            self.process_queue.start()
            logger.info(f"✅ Queue processor started for channel {self.channel_id}")
        else:
            logger.warning("⚠️ DISCORD_CHANNEL_ID not set - notifications disabled")
    
    @tasks.loop(seconds=2)
    async def process_queue(self):
        """Process message queue and send messages"""
        try:
            while not self.message_queue.empty():
                try:
                    item = self.message_queue.get_nowait()
                    
                    # Handle tuple (embed, file) or single embed
                    if isinstance(item, tuple):
                        embed, file = item
                    else:
                        embed, file = item, None
                        
                    channel = self.bot.get_channel(self.channel_id)
                    if channel:
                        try:
                            if file:
                                await channel.send(embed=embed, file=file)
                            else:
                                await channel.send(embed=embed)
                            logger.debug(f"✅ Sent message to {channel.name}")
                        except discord.errors.Forbidden:
                            logger.error(f"❌ No permission to send message in channel {self.channel_id}")
                        except Exception as e:
                            logger.error(f"❌ Failed to send message: {e}")
                    else:
                        logger.error(f"❌ Channel {self.channel_id} not found")
                except Exception as e:
                    logger.error(f"Error getting from queue: {e}")
                    break
        except Exception as e:
            logger.error(f"❌ Error processing message queue: {e}")
    
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
    
    @commands.command(name='recommendation', help='Get latest trading recommendations')
    async def recommendation_command(self, ctx):
        """Get latest trading recommendations"""
        try:
            embed = discord.Embed(
                title="📊 Latest Trading Signals",
                description="All available trading signals",
                color=discord.Color.blue()
            )
            
            if not GLOBAL_SIGNALS:
                embed.add_field(name="Status", value="No signals available yet", inline=False)
            else:
                # BUY signals
                buy_signals = [s for s in GLOBAL_SIGNALS.values() if 'BUY' in s.get('signal_type', '')]
                if buy_signals:
                    buy_text = ""
                    for sig in buy_signals[:5]:  # Show top 5
                        symbol = sig.get('symbol', 'N/A')
                        confidence = sig.get('confidence', 0)
                        buy_text += f"**{symbol}**: {confidence:.0%}\n"
                    embed.add_field(name="🟢 BUY Signals", value=buy_text[:1024], inline=True)
                
                # SELL signals
                sell_signals = [s for s in GLOBAL_SIGNALS.values() if 'SELL' in s.get('signal_type', '')]
                if sell_signals:
                    sell_text = ""
                    for sig in sell_signals[:5]:  # Show top 5
                        symbol = sig.get('symbol', 'N/A')
                        confidence = sig.get('confidence', 0)
                        sell_text += f"**{symbol}**: {confidence:.0%}\n"
                    embed.add_field(name="🔴 SELL Signals", value=sell_text[:1024], inline=True)
                
                # NEUTRAL signals
                neutral_signals = [s for s in GLOBAL_SIGNALS.values() if 'NEUTRAL' in s.get('signal_type', '')]
                if neutral_signals:
                    neutral_text = f"{len(neutral_signals)} cryptocurrencies"
                    embed.add_field(name="⚪ NEUTRAL", value=neutral_text, inline=True)
            
            embed.set_footer(text=f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
            await ctx.send(embed=embed)
            logger.info(f"✅ Recommendation command executed by {ctx.author}")
        except Exception as e:
            logger.error(f"Error in recommendation command: {e}")
            await ctx.send(f"❌ Error: {e}")
    
    @commands.command(name='portfolio', help='View full portfolio status')
    async def portfolio_command(self, ctx):
        """View full portfolio status for all cryptocurrencies"""
        try:
            embed = discord.Embed(
                title="💰 Portfolio Overview",
                description="Status of all monitored cryptocurrencies",
                color=discord.Color.purple()
            )
            
            if not GLOBAL_SIGNALS:
                embed.add_field(name="Status", value="No data available yet", inline=False)
            else:
                # Summary statistics
                total_coins = len(GLOBAL_SIGNALS)
                buy_count = sum(1 for s in GLOBAL_SIGNALS.values() if 'BUY' in s.get('signal_type', ''))
                sell_count = sum(1 for s in GLOBAL_SIGNALS.values() if 'SELL' in s.get('signal_type', ''))
                neutral_count = sum(1 for s in GLOBAL_SIGNALS.values() if 'NEUTRAL' in s.get('signal_type', ''))
                
                # FIX: 優先使用 AI Validity 作為信心度指標，如果沒有則使用原始 confidence
                def get_display_confidence(sig):
                    ai_score = sig.get('ai_validity')
                    if ai_score is not None:
                        return float(ai_score) / 100.0
                    return sig.get('confidence', 0)

                avg_confidence = sum(get_display_confidence(s) for s in GLOBAL_SIGNALS.values()) / total_coins if total_coins > 0 else 0
                
                embed.add_field(name="👮 Market Status", 
                               value=f"Monitoring {total_coins} coins\nAvg AI Score: {avg_confidence:.1%}",
                               inline=True)
                embed.add_field(name="🟢 Buy Signals", value=f"{buy_count}", inline=True)
                embed.add_field(name="🔴 Sell Signals", value=f"{sell_count}", inline=True)
                
                # Create detailed signal table
                signals_table = "```\nSymbol  | Signal        | AI Score | Trend\n"
                signals_table += "--------|---------------|----------|-------------------\n"
                
                for symbol in sorted(GLOBAL_SIGNALS.keys()):
                    sig = GLOBAL_SIGNALS[symbol]
                    signal = sig.get('signal_type', 'N/A')[:12].ljust(12)
                    
                    # 顯示 AI 分數
                    ai_score = sig.get('ai_validity')
                    if ai_score is not None:
                        conf_str = f"{float(ai_score):.0f}".ljust(8)  # 顯示整數分數 (e.g., 60)
                    else:
                        conf_str = f"{sig.get('confidence', 0):.0%}".ljust(8) # 顯示原始信心 (e.g., 50%)
                        
                    trend = sig.get('trend_direction', 'N/A')[:15].ljust(15)
                    signals_table += f"{symbol:6} | {signal} | {conf_str} | {trend}\n"
                    if len(signals_table) > 1900:  # Leave room for closing
                        signals_table = signals_table[:1900] + "...\n```"
                        break
                else:
                    signals_table += "```"
                
                embed.add_field(name="📊 Signal Details", value=signals_table, inline=False)
            
            embed.set_footer(text=f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            await ctx.send(embed=embed)
            logger.info(f"✅ Portfolio command executed by {ctx.author}")
        except Exception as e:
            logger.error(f"Error in portfolio command: {e}")
            await ctx.send(f"❌ Error: {str(e)[:100]}")
    
    @commands.command(name='symbol', help='Get signal for specific symbol')
    async def symbol_command(self, ctx, symbol: str):
        """Get detailed signal for a specific symbol"""
        try:
            symbol = symbol.upper()
            sig = GLOBAL_SIGNALS.get(symbol)
            
            if not sig:
                await ctx.send(f"❌ Symbol {symbol} not found")
                return
            
            # Determine color based on signal
            if 'BUY' in sig.get('signal_type', ''):
                color = discord.Color.green()
            elif 'SELL' in sig.get('signal_type', ''):
                color = discord.Color.red()
            else:
                color = discord.Color.yellow()
            
            embed = discord.Embed(
                title=f"{sig.get('signal_type', 'N/A')} {symbol}USDT",
                color=color,
                timestamp=datetime.now()
            )
            
            # Safe float conversion helper
            def safe_float(val):
                try:
                    return float(val) if val is not None else 0.0
                except (ValueError, TypeError):
                    return 0.0

            current_price = safe_float(sig.get('current_price'))
            predicted_price = safe_float(sig.get('predicted_price'))
            confidence = safe_float(sig.get('confidence'))
            rsi = safe_float(sig.get('rsi'))
            entry_price = safe_float(sig.get('entry_price'))
            take_profit = safe_float(sig.get('take_profit'))
            stop_loss = safe_float(sig.get('stop_loss'))
            
            # Calculate price change safely
            if current_price > 0:
                price_change = (predicted_price - current_price) / current_price * 100
            else:
                price_change = 0.0
            
            embed.add_field(name="💲 Current Price", value=f"${current_price:,.2f}", inline=True)
            embed.add_field(name="🎯 Predicted Price", value=f"${predicted_price:,.2f}", inline=True)
            embed.add_field(name="📈 Price Change", value=f"{price_change:+.2f}%", inline=True)
            
            embed.add_field(name="💯 Confidence", value=f"{confidence:.1%}", inline=True)
            embed.add_field(name="📉 Trend", value=str(sig.get('trend_direction', 'N/A')), inline=True)
            embed.add_field(name="📊 RSI", value=f"{rsi:.1f}", inline=True)
            
            embed.add_field(name="🎶 Entry", value=f"${entry_price:,.2f}", inline=True)
            embed.add_field(name="✅ TP", value=f"${take_profit:,.2f}", inline=True)
            embed.add_field(name="❌ SL", value=f"${stop_loss:,.2f}", inline=True)
            
            # Add AI Analysis info if available
            ai_validity = sig.get('ai_validity')
            if ai_validity is not None:
                embed.add_field(name="🤖 AI Score", value=f"{float(ai_validity):.0f}/100", inline=True)

            embed.set_footer(text=f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            await ctx.send(embed=embed)
            logger.info(f"✅ Symbol command for {symbol} executed by {ctx.author}")
        except Exception as e:
            logger.error(f"Error in symbol command: {e}", exc_info=True)
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
            embed.add_field(name="Monitored Coins", value=f"{len(GLOBAL_SIGNALS)}", inline=True)
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
    
    def queue_embed(self, embed: discord.Embed, file: discord.File = None):
        """Queue an embed (and optional file) to be sent (thread-safe)"""
        if self.training_cog:
            try:
                if file:
                    self.training_cog.message_queue.put((embed, file))
                else:
                    self.training_cog.message_queue.put(embed)
                logger.debug(f"✅ Message queued (queue size: {self.training_cog.message_queue.qsize()})")
            except Exception as e:
                logger.error(f"❌ Error queueing message: {e}")
        else:
            logger.warning("⚠️ TrainingNotificationCog not available yet, message discarded")
    
    def update_signal(self, symbol: str, signal_data: dict):
        """Update signal data in global storage (thread-safe)"""
        try:
            GLOBAL_SIGNALS[symbol] = signal_data
            logger.debug(f"✅ Updated signal for {symbol}")
        except Exception as e:
            logger.error(f"Error updating signal: {e}")
    
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
