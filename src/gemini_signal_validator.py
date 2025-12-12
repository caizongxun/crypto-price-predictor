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
    使用 Groq API 驗證交易信號 (JSON Mode)
    """
    
    def __init__(self, api_key: str):
        self.api_key = os.getenv('GROQ_API_KEY') or api_key
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
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
            # 構建提示詞，加入評分指南以鼓勵更合理的給分
            prompt = f"""You are a Growth-Oriented Crypto Analyst. Analyze this potential trade setup.

Signal Data:
- Symbol: {symbol}
- Price: ${current_price:,.2f}
- Signal: {signal_type} (Model Confidence: {confidence:.1f}%)

Technical Context:
- RSI: {technical_indicators.get('rsi', 'N/A')} (Consider oversold < 30, overbought > 70)
- MACD: {technical_indicators.get('macd', 'N/A')}
- Trends (1H/4H/1D): {short_term_analysis.get('trend')} / {medium_term_analysis.get('trend')} / {long_term_analysis.get('trend')}

Scoring Guide:
- 80-100: Excellent setup (Strong trend alignment + good indicators)
- 60-79: Good potential (Some mixed signals but overall positive structure)
- 40-59: Weak/Neutral (Sideways or conflicting signals, only take if low risk)
- 0-39: Bad setup (Counter-trend or dangerous)

Instruction:
- Even if the trend is NEUTRAL, look for reversal signs or consolidation breakouts.
- Don't be too conservative. If RSI is good or trend is starting, give at least 60-65.
- Output strictly in JSON.

Output JSON structure:
{{
    "validity_score": (float 0-100),
    "entry_offset_pct": (float, e.g. -0.2),
    "stop_loss_offset_pct": (float, e.g. 2.0),
    "take_profit_offset_pct": (float, e.g. 5.0),
    "market_condition": ("Bullish", "Bearish", "Sideways"),
    "confidence_adjustment": (float, -30 to +30),
    "reasoning": (string, max 15 words)
}}
"""

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a savvy trading assistant looking for opportunities. Output strictly valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3, # 稍微提高溫度，允許更多可能性
                "response_format": {"type": "json_object"},
                "max_tokens": 500
            }

            response = requests.post(self.api_url, headers=headers, json=data, timeout=10)
            
            if response.status_code != 200:
                logger.error(f"❌ Groq API Error {response.status_code}: {response.text}")
                return self._create_default_analysis(signal_type, confidence)
            
            result = response.json()
            response_text = result['choices'][0]['message']['content']
            
            return self._parse_json_response(response_text, signal_type, confidence, current_price)
        
        except Exception as e:
            logger.error(f"❌ Groq 分析異常 ({symbol}): {e}")
            return self._create_default_analysis(signal_type, confidence)
    
    def _parse_json_response(
        self,
        json_text: str,
        signal_type: str,
        confidence: float,
        current_price: float
    ) -> GeminiAnalysis:
        try:
            data = json.loads(json_text)
            
            validity_score = float(data.get("validity_score", confidence))
            entry_offset = float(data.get("entry_offset_pct", 0.0))
            sl_offset = float(data.get("stop_loss_offset_pct", 2.0))
            tp_offset = float(data.get("take_profit_offset_pct", 5.0))
            conf_adj = float(data.get("confidence_adjustment", 0.0))
            
            # 計算實際價格
            entry_price = current_price * (1 + entry_offset / 100)
            
            if "BUY" in signal_type:
                stop_loss = current_price * (1 - sl_offset / 100)
                take_profit = current_price * (1 + tp_offset / 100)
            else:
                stop_loss = current_price * (1 + sl_offset / 100)
                take_profit = current_price * (1 - tp_offset / 100)
            
            # 計算盈虧比
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            rr_ratio = reward / risk if risk > 0 else 0
            
            return GeminiAnalysis(
                is_valid=validity_score >= 60,
                validity_score=validity_score,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=rr_ratio,
                reasoning=data.get("reasoning", "AI Analysis"),
                market_condition=data.get("market_condition", "Sideways"),
                confidence_adjustment=conf_adj
            )
            
        except Exception as e:
            logger.error(f"❌ JSON 解析失敗: {e} | Raw: {json_text[:100]}")
            return self._create_default_analysis(signal_type, confidence)

    @staticmethod
    def _create_default_analysis(signal_type: str, confidence: float) -> GeminiAnalysis:
        return GeminiAnalysis(
            is_valid=confidence > 65,
            validity_score=confidence,
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            risk_reward_ratio=None,
            reasoning="AI 服務暫時不可用",
            market_condition="未知",
            confidence_adjustment=0
        )
