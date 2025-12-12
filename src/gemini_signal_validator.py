import os
import logging
import re
import json
import requests
from typing import Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GeminiAnalysis:
    """AI 分析結果"""
    is_valid: bool
    validity_score: float
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    risk_reward_ratio: Optional[float]
    reasoning: str
    market_condition: str
    confidence_adjustment: float

class GeminiSignalValidator:
    """
    使用 Groq API 驗證交易信號
    (使用 requests 直接調用 API，避免庫版本衝突)
    """
    
    def __init__(self, api_key: str):
        self.api_key = os.getenv('GROQ_API_KEY') or api_key
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        # 更新為最新的 Groq 推薦模型
        self.model = "llama-3.3-70b-versatile"
        
        if not self.api_key:
            logger.error("❌ 未設置 GROQ_API_KEY")
            self.enabled = False
        else:
            self.enabled = True
            logger.info(f"✅ Groq AI 已就緒 (Direct API: {self.model})")
    
    def validate_signal(
        self,
        symbol: str,
        signal_type: str,
        confidence: float,
        current_price: float,
        short_term_analysis: Dict,
        medium_term_analysis: Dict,
        long_term_analysis: Dict,
        technical_indicators: Dict,
        market_context: str = ""
    ) -> Optional[GeminiAnalysis]:
        
        if not self.enabled:
            return self._create_default_analysis(signal_type, confidence)
        
        try:
            prompt = f"""你是專業的加密貨幣交易分析師。請分析以下交易信號並給出 JSON 格式的建議。

【交易數據】
- 幣種: {symbol}
- 價格: ${current_price:,.2f}
- 信號: {signal_type} (原始信心: {confidence:.1f}%)

【趨勢分析】
- 1H (短線): {short_term_analysis.get('trend')}
- 4H (中線): {medium_term_analysis.get('trend')}
- 1D (長線): {long_term_analysis.get('trend')}

【技術指標】
- RSI: {technical_indicators.get('rsi', 'N/A')}
- MACD: {technical_indicators.get('macd', 'N/A')}
- Volume: {technical_indicators.get('volume_trend', 'N/A')}

請嚴格按照以下格式回答（不要有廢話，只回數字和簡短理由）：

1. 有效性評分 (0-100)
2. 建議進場位偏移% (例如 -0.5 代表低於現價 0.5% 進場)
3. 止損位偏移% (例如 2.0 代表風險 2%)
4. 止盈位偏移% (例如 5.0 代表目標 5%)
5. 市場狀態 (牛市/熊市/盤整)
6. 信心調整值 (-30 到 +30)
7. 理由 (一句話)"""

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一個嚴謹的量化交易員。只輸出關鍵數據，不輸出閒聊。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.5,
                "max_tokens": 1000
            }

            response = requests.post(self.api_url, headers=headers, json=data, timeout=10)
            
            # 如果失敗，記錄詳細錯誤訊息
            if response.status_code != 200:
                logger.error(f"❌ Groq API Error {response.status_code}: {response.text}")
                response.raise_for_status()
            
            result = response.json()
            response_text = result['choices'][0]['message']['content']
            
            return self._parse_response(response_text, signal_type, confidence, current_price)
        
        except Exception as e:
            logger.error(f"❌ Groq 分析失敗 ({symbol}): {e}")
            return self._create_default_analysis(signal_type, confidence)
    
    def _parse_response(
        self,
        response_text: str,
        signal_type: str,
        confidence: float,
        current_price: float
    ) -> GeminiAnalysis:
        try:
            # 使用正則表達式提取數值
            validity_score = self._extract_number(response_text, ["評分", "score", "validity"], default=confidence)
            entry_adjustment = self._extract_number(response_text, ["進場", "entry"], default=0.0)
            stop_loss_pct = self._extract_number(response_text, ["止損", "stop", "sl"], default=2.0)
            take_profit_pct = self._extract_number(response_text, ["止盈", "profit", "tp"], default=5.0)
            confidence_adjustment = self._extract_number(response_text, ["調整", "adjust"], default=0.0)
            
            # 判斷是否有效
            is_valid = validity_score >= 60
            
            # 計算實際價格
            entry_price = current_price * (1 + entry_adjustment / 100)
            
            if signal_type in ["BUY", "STRONG_BUY"]:
                stop_loss = current_price * (1 - stop_loss_pct / 100)
                take_profit = current_price * (1 + take_profit_pct / 100)
            else:
                stop_loss = current_price * (1 + stop_loss_pct / 100)
                take_profit = current_price * (1 - take_profit_pct / 100)
            
            # 計算盈虧比
            potential_loss = abs(entry_price - stop_loss)
            potential_gain = abs(take_profit - entry_price)
            risk_reward = potential_gain / potential_loss if potential_loss > 0 else 0
            
            # 提取市場狀態
            market_condition = "盤整"
            if "牛市" in response_text or "bullish" in response_text.lower():
                market_condition = "牛市"
            elif "熊市" in response_text or "bearish" in response_text.lower():
                market_condition = "熊市"
            
            # 提取理由（移除數字行，只留文字）
            lines = response_text.split('\n')
            reasoning = next((line for line in reversed(lines) if len(line) > 10 and not any(c.isdigit() for c in line)), "AI 綜合技術指標分析")
            
            return GeminiAnalysis(
                is_valid=is_valid,
                validity_score=max(0, min(100, validity_score)),
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=risk_reward,
                reasoning=reasoning[:200],
                market_condition=market_condition,
                confidence_adjustment=max(-30, min(30, confidence_adjustment))
            )
        
        except Exception as e:
            logger.error(f"❌ 解析 Groq 回應失敗: {e}")
            return self._create_default_analysis(signal_type, confidence)
    
    @staticmethod
    def _extract_number(text: str, keywords: list, default: float = 0) -> float:
        for line in text.split('\n'):
            for keyword in keywords:
                if keyword in line.lower():
                    # 尋找行內的數字
                    matches = re.findall(r'-?\d+(?:\.\d+)?', line)
                    if matches:
                        return float(matches[-1])
        return default
    
    @staticmethod
    def _create_default_analysis(signal_type: str, confidence: float) -> GeminiAnalysis:
        return GeminiAnalysis(
            is_valid=confidence > 65,
            validity_score=confidence,
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            risk_reward_ratio=None,
            reasoning="AI 分析服務暫時不可用",
            market_condition="未知",
            confidence_adjustment=0
        )
