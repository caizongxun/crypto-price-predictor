import os
import discord
from discord.ext import commands, tasks
import asyncio
from datetime import datetime
import logging
from pathlib import Path
import sys

# 新夁 sys.path
sys.path.insert(0, str(Path(__file__).parent))

from model_trainer import ModelTrainer
from data_fetcher import DataFetcher

logger = logging.getLogger(__name__)

class DiscordBotHandler:
    """
    Discord Bot 處理器 - 管理所有 Discord 相關的功能
    """
    
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.data_fetcher = None
        self.model_trainer = None
        self.models = {}  # 存储已加載的模型
        self.predictions_cache = {}  # 上次預測了別了
        self.last_update = {}
        
        logger.info("DiscordBotHandler initialized")
    
    async def initialize(self):
        """初始化处理器"""
        try:
            self.data_fetcher = DataFetcher()
            self.model_trainer = ModelTrainer()
            logger.info("Handler initialization complete")
        except Exception as e:
            logger.error(f"Handler initialization failed: {e}")
            raise
    
    async def get_prediction(self, symbol: str) -> dict:
        """
        獲取加密貨幣价格預測
        
        Args:
            symbol: 加密貨幣符號 (e.g., 'BTC', 'ETH')
            
        Returns:
            預測結果字典
        """
        try:
            # 檢查緩存
            if symbol in self.predictions_cache:
                cache_data = self.predictions_cache[symbol]
                if (datetime.now() - cache_data['timestamp']).total_seconds() < 300:  # 5分鐘产效期
                    return cache_data['prediction']
            
            # 獲取新數據
            logger.info(f"Fetching data for {symbol}...")
            data = await self.data_fetcher.fetch_data(symbol)
            
            if not data:
                return {"error": f"Failed to fetch data for {symbol}"}
            
            # 模型預測
            logger.info(f"Getting prediction for {symbol}...")
            prediction = await self.model_trainer.predict(symbol, data)
            
            # 緩存結果
            self.predictions_cache[symbol] = {
                'prediction': prediction,
                'timestamp': datetime.now()
            }
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error getting prediction for {symbol}: {e}")
            return {"error": str(e)}
    
    def format_prediction_embed(self, symbol: str, prediction: dict) -> discord.Embed:
        """
        格式化預測為 Discord Embed
        
        Args:
            symbol: 加密貨幣符號
            prediction: 預測結果字典
            
        Returns:
            Discord Embed 物件
        """
        if "error" in prediction:
            embed = discord.Embed(
                title=f"Prediction Error - {symbol}",
                description=prediction["error"],
                color=discord.Color.red()
            )
            return embed
        
        # 繊建成功預測的 Embed
        price = prediction.get('predicted_price', 'N/A')
        confidence = prediction.get('confidence', 'N/A')
        trend = prediction.get('trend', 'N/A')
        
        # 根據趨务選擇颜色
        color = discord.Color.green() if trend == "UP" else discord.Color.red()
        
        embed = discord.Embed(
            title=f"{symbol} Price Prediction",
            description=f"Predicted Price: ${price}",
            color=color
        )
        
        embed.add_field(
            name="Trend",
            value=f"{trend} 📈" if trend == "UP" else f"{trend} 📉",
            inline=False
        )
        embed.add_field(
            name="Confidence",
            value=f"{confidence}%",
            inline=False
        )
        embed.add_field(
            name="Timestamp",
            value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            inline=False
        )
        
        return embed
    
    @tasks.loop(hours=1)
    async def update_bot_status(self):
        """定時更新 Bot 的程徏狀態"""
        try:
            statuses = [
                "Bitcoin prices 📊",
                "Ethereum updates 🔨",
                "Crypto trends 💮",
                "!help_crypto for commands 🔢"
            ]
            
            status_index = len(self.bot.guilds) % len(statuses)
            await self.bot.change_presence(
                activity=discord.Activity(
                    type=discord.ActivityType.watching,
                    name=statuses[status_index]
                )
            )
        except Exception as e:
            logger.error(f"Error updating bot status: {e}")
    
    async def handle_error(self, ctx: commands.Context, error: Exception):
        """
        處理鄙誤
        
        Args:
            ctx: 上下文
            error: 鄙誤物件
        """
        logger.error(f"Command error: {error}")
        
        embed = discord.Embed(
            title="Error Occurred",
            description=str(error),
            color=discord.Color.red()
        )
        
        try:
            await ctx.send(embed=embed)
        except:
            await ctx.send(f"Error: {error}")
    
    def setup_commands(self):
        """設定所有命令"""
        
        @self.bot.command(name='predict', help='Get price prediction')
        async def predict(ctx, symbol: str):
            """Get cryptocurrency price prediction"""
            async with ctx.typing():
                prediction = await self.get_prediction(symbol.upper())
                embed = self.format_prediction_embed(symbol.upper(), prediction)
                await ctx.send(embed=embed)
        
        @self.bot.command(name='stats', help='Get bot statistics')
        async def stats(ctx):
            """Show bot statistics"""
            embed = discord.Embed(
                title="Bot Statistics",
                color=discord.Color.blue()
            )
            
            embed.add_field(
                name="Guilds",
                value=str(len(self.bot.guilds)),
                inline=True
            )
            embed.add_field(
                name="Cached Predictions",
                value=str(len(self.predictions_cache)),
                inline=True
            )
            embed.add_field(
                name="Uptime",
                value="Running",
                inline=True
            )
            
            await ctx.send(embed=embed)

def setup_handler(bot: commands.Bot) -> DiscordBotHandler:
    """設定並傳回处理器實高"""
    handler = DiscordBotHandler(bot)
    handler.setup_commands()
    return handler
