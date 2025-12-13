import os
import discord
from discord.ext import commands
import asyncio
from dotenv import load_dotenv
import logging

# 載入環境變數
load_dotenv()

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 建立 Bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# 當 Bot 啟動
@bot.event
async def on_ready():
    logger.info(f'{bot.user} has connected to Discord!')
    logger.info(f'Bot is in {len(bot.guilds)} guild(s)')
    
    # 設定 Bot 狀態
    await bot.change_presence(
        activity=discord.Activity(
            type=discord.ActivityType.watching,
            name='Crypto Prices 📊'
        )
    )

# 當收到訊息
@bot.event
async def on_message(message):
    # 忽略 Bot 自己的訊息
    if message.author == bot.user:
        return
    
    # 處理命令
    await bot.process_commands(message)

# 健康檢查命令
@bot.command(name='health', help='Check bot health status')
async def health_check(ctx):
    """檢查 Bot 狀態"""
    embed = discord.Embed(
        title="Bot Health Status ✓",
        description="Bot is online and functioning properly",
        color=discord.Color.green()
    )
    embed.add_field(name="Latency", value=f"{bot.latency * 1000:.2f}ms", inline=False)
    embed.add_field(name="Guilds", value=f"{len(bot.guilds)}", inline=False)
    await ctx.send(embed=embed)

# 幫助命令
@bot.command(name='help_crypto', help='Show crypto prediction help')
async def help_crypto(ctx):
    """顯示加密貨幣預測幫助"""
    embed = discord.Embed(
        title="Crypto Price Predictor Bot",
        description="Commands for cryptocurrency price prediction",
        color=discord.Color.blue()
    )
    
    embed.add_field(
        name="!predict <symbol>",
        value="Get price prediction for a cryptocurrency (e.g., !predict BTC)",
        inline=False
    )
    embed.add_field(
        name="!health",
        value="Check bot health status",
        inline=False
    )
    embed.add_field(
        name="!supported",
        value="List supported cryptocurrencies",
        inline=False
    )
    
    await ctx.send(embed=embed)

# 支持的加密貨幣命令
@bot.command(name='supported', help='List supported cryptocurrencies')
async def supported_cryptos(ctx):
    """列出支持的加密貨幣"""
    supported = [
        "Bitcoin (BTC)",
        "Ethereum (ETH)",
        "More coins coming soon..."
    ]
    
    embed = discord.Embed(
        title="Supported Cryptocurrencies",
        description="\n".join(supported),
        color=discord.Color.orange()
    )
    
    await ctx.send(embed=embed)

def run_bot():
    """執行 Bot"""
    token = os.getenv('DISCORD_BOT_TOKEN')
    
    if not token:
        logger.error("DISCORD_BOT_TOKEN not found in environment variables!")
        raise ValueError("DISCORD_BOT_TOKEN is required")
    
    try:
        bot.run(token)
    except Exception as e:
        logger.error(f"Error running bot: {e}")
        raise

if __name__ == "__main__":
    run_bot()
