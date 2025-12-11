# 加密货币价格预测系统 🚀

高级加密货币价格预测系统，使用LSTM和Transformer模型，支持实时Discord通知。

## 功能特性

✨ **实时价格预测** - 使用深度学习模型预测加密货币价格
📊 **多币种监控** - 同时监控15-20+种加密货币
🎯 **交易信号** - 识别支撑/阻力水平，提供买入/卖出建议
🤖 **Discord机器人** - 自动推送交易信号通知
💰 **真实数据** - 使用币安、CoinGecko等真实API数据
⚙️ **生产就绪** - 完全配置用于连续监控

## 支持的加密货币

- 比特币 (BTC)
- 以太坊 (ETH)
- 币安币 (BNB)
- 索拉纳 (SOL)
- XRP
- 卡尔达诺 (ADA)
- 狗狗币 (DOGE)
- 波卡 (DOT)
- 雪崩 (AVAX)
- 多边形 (MATIC)
- 莱特币 (LTC)
- 链接 (LINK)
- 合成器 (UNI)
- Aave (AAVE)
- 复合 (COMP)
- Yearn Finance (YFI)
- Arbitrum (ARB)
- Optimism (OP)

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/crypto-price-predictor.git
cd crypto-price-predictor
```

### 2. 环境设置

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. 配置

```bash
cp .env.example .env
# 编辑 .env 添加你的 Discord bot token
```

### 4. 运行

```bash
python main.py
```

## 项目结构

```
crypto-price-predictor/
├── src/
│   ├── data_fetcher.py          # 数据获取
│   ├── model_trainer.py         # 模型训练
│   ├── predictor.py             # 价格预测
│   ├── technical_analysis.py    # 技术分析
│   ├── discord_bot.py           # Discord机器人
│   └── utils.py                 # 工具函数
├── config/
│   └── config.yaml              # 配置文件
├── models/
│   └── saved_models/            # 已训练模型
├── data/
│   └── historical/              # 历史数据
├── main.py                      # 主程序
├── train_model.py               # 训练脚本
├── requirements.txt             # 依赖
└── README.md
```

## 模型详情

### LSTM模型
- 双向LSTM + 注意力机制
- 60天历史数据输入
- 7天价格预测
- 准确度 75%+

### Transformer模型
- 多头自注意力
- HuggingFace预训练
- 可以微调

## API数据源

- **币安 (Binance)** - 实时OHLCV数据
- **CoinGecko** - 市场数据
- **Kraken** - 专业级数据

## 配置说明

编辑 `config/config.yaml`:

```yaml
# Discord配置
discord:
  enabled: true
  update_interval: 3600  # 1小时

# 监控的加密货币
cryptocurrencies:
  - symbol: "BTC"
    trading_pair: "BTCUSDT"
  - symbol: "ETH"
    trading_pair: "ETHUSDT"

# 模型配置
model:
  type: "lstm"  # 或 "transformer"
  lookback_period: 60
  prediction_horizon: 7
  epochs: 100
```

## 训练模型

```bash
# 训练LSTM模型
python train_model.py --symbol BTC --model lstm --epochs 100

# 训练Transformer模型
python train_model.py --symbol BTC --model transformer --epochs 100
```

## Discord机器人设置

### 1. 创建Discord应用
1. 访问 [Discord开发者门户](https://discord.com/developers/applications)
2. 点击"New Application"
3. 进入"Bot"部分，点击"Add Bot"
4. 复制Token到.env文件

### 2. 获取频道ID
1. 启用开发者模式
2. 右键频道 → 复制频道ID
3. 添加到.env

## 交易信号说明

```
BTC 交易信号
├─ 当前价格: $45,000
├─ 预测价格 (7天): $48,500 (+7.8%)
├─ 置信度: 82%
├─ 支撑位: $43,200
├─ 阻力位: $49,800
├─ 入场建议: $44,000-$44,500
├─ 止盈: $48,000 | $49,500 | $51,000
└─ 止损: $43,000
```

## Docker部署

### 使用Docker Compose

```bash
# 启动
docker-compose up -d

# 查看日志
docker-compose logs -f crypto-predictor

# 停止
docker-compose down
```

## 免责声明

⚠️ **重要提示**:
1. 预测是概率性的，不保证准确
2. 过去表现≠未来结果
3. 进行自己的研究 (DYOR)
4. 本工具仅供信息参考
5. 不是财务建议 - 咨询财务顾问
6. 加密货币具有高风险

## 常见问题

### 模型准确度低
- 增加回看周期到90+天
- 添加更多技术指标
- 用最新数据重新训练

### Discord连接失败
- 验证bot token正确
- 检查频道ID
- 确认bot有发送消息权限

### 数据获取错误
- 验证API密钥
- 检查API速率限制
- 确保网络连接

## 支持与贡献

欢迎提交问题和拉取请求！

## 许可证

MIT许可证 - 查看LICENSE文件

## 更新日期

2024年12月
