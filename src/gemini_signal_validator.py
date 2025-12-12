import google.generativeai as genai
import logging
import re
from typing import Dict, Optional
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class GeminiAnalysis:
    """Gemini AI 分析結果"""
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
    使用 Gemini API 驗證交易信號
    """
    
    def __init__(self, api_key: str):
        """
        初始化 Gemini 驗證器
        """
        try:
            genai.configure(api_key=api_key)
            # 404 error fix: 恢復使用 gemini-2.0-flash-exp (因為 1.5-flash 在目前環境不可用)
            # 同時增加 retry 機制處理 429 配額問題
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            logger.info("✅ Gemini 2.0 Flash 已連接")
        except Exception as e:
            logger.error(f"❌ Gemini 連接失敗: {e}")
            self.model = None
    
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
        
        if not self.model:
            return self._create_default_analysis(signal_type, confidence)
        
        try:
            # 增加較長的延遲以避免 429 Rate Limit
            time.sleep(4) 
            
            prompt = f"""你是專業的加密貨幣交易分析師。請快速分析這個交易信號：

【交易信息】
- 符號: {symbol}
- 當前價格: ${current_price:,.2f}
- 信號: {signal_type}
- 原始信心度: {confidence:.1f}%

【多時間框架分析】
短期(1h): {short_term_analysis.get('trend', 'N/A')} ({short_term_analysis.get('confidence', 0):.0f}%)
中期(4h): {medium_term_analysis.get('trend', 'N/A')} ({medium_term_analysis.get('confidence', 0):.0f}%)
長期(1d): {long_term_analysis.get('trend', 'N/A')} ({long_term_analysis.get('confidence', 0):.0f}%)

【技術指標】
- RSI: {technical_indicators.get('rsi', 'N/A')}
- MACD: {technical_indicators.get('macd', 'N/A')}
- 成交量趨勢: {technical_indicators.get('volume_trend', 'N/A')}

{f"【市場背景】{market_context}" if market_context else ""}

請用數字格式回答（簡潔）：
1. 信號有效性？(是/否) + 評分(0-100)
2. 建議進場價格偏移(%)：相對當前價
3. 止損百分比(%)：向下風險
4. 止盈百分比(%)：向上目標
5. 市場狀態：牛市/熊市/盤整
6. 信心度調整：-30 到 +30 之間
7. 理由(一句話)"""
            
            # 嘗試調用 API
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config={
                        'temperature': 0.7,
                        'top_p': 0.95,
                        'top_k': 40,
                        'max_output_tokens': 500,
                    }
                )
                return self._parse_gemini_response(
                    response.text,
                    signal_type,
                    confidence,
                    current_price
                )
            except Exception as api_error:
                # 如果是配額問題 (429)，等待後重試一次
                if "429" in str(api_error) or "Quota exceeded" in str(api_error):
                    logger.warning(f"⚠️ 配額滿，等待 10 秒後重試 {symbol}...")
                    time.sleep(10)
                    response = self.model.generate_content(prompt)
                    return self._parse_gemini_response(
                        response.text,
                        signal_type,
                        confidence,
                        current_price
                    )
                else:
                    raise api_error

        except Exception as e:
            # 記錄具體錯誤
            error_msg = str(e)
            if "429" in error_msg:
                logger.warning(f"⚠️ Gemini 配額耗盡 ({symbol})，跳過 AI 驗證")
            elif "404" in error_msg:
                logger.error(f"❌ Gemini 模型未找到 ({symbol}): 請檢查 API Key 或模型名稱")
            else:
                logger.error(f"❌ Gemini 分析失敗 ({symbol}): {e}")
            
            return self._create_default_analysis(signal_type, confidence)
    
    def _parse_gemini_response(
        self,
        response_text: str,
        signal_type: str,
        confidence: float,
        current_price: float
    ) -> GeminiAnalysis:
        try:
            is_valid = "是" in response_text or "valid" in response_text.lower()
            
            validity_score = self._extract_number(
                response_text,
                ["評分", "評分:", "score", "有效性"],
                default=confidence
            )
            
            entry_adjustment = self._extract_number(
                response_text,
                ["進場", "entry", "偏移"],
                default=0.0
            )
            
            stop_loss_pct = self._extract_number(
                response_text,
                ["止損", "stop", "risk"],
                default=2.0
            )
            
            take_profit_pct = self._extract_number(
                response_text,
                ["止盈", "profit", "target", "目標"],
                default=5.0
            )
            
            confidence_adjustment = self._extract_number(
                response_text,
                ["調整", "adjust"],
                default=0.0
            )
            
            entry_price = current_price * (1 + entry_adjustment / 100)
            
            if signal_type in ["BUY", "STRONG_BUY"]:
                stop_loss = current_price * (1 - stop_loss_pct / 100)
                take_profit = current_price * (1 + take_profit_pct / 100)
            else:
                stop_loss = current_price * (1 + stop_loss_pct / 100)
                take_profit = current_price * (1 - take_profit_pct / 100)
            
            potential_loss = abs(current_price - stop_loss)
            potential_gain = abs(take_profit - current_price)
            risk_reward = potential_gain / potential_loss if potential_loss > 0 else 0
            
            market_condition = "盤整"
            if "牛市" in response_text or "bullish" in response_text.lower():
                market_condition = "牛市"
            elif "熊市" in response_text or "bearish" in response_text.lower():
                market_condition = "熊市"
            
            return GeminiAnalysis(
                is_valid=is_valid,
                validity_score=max(0, min(100, validity_score)),
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=risk_reward,
                reasoning=response_text[:400],
                market_condition=market_condition,
                confidence_adjustment=max(-30, min(30, confidence_adjustment))
            )
        
        except Exception as e:
            logger.error(f"❌ 解析 Gemini 回應失敗: {e}")
            return self._create_default_analysis(signal_type, confidence)
    
    @staticmethod
    def _extract_number(text: str, keywords: list, default: float = 0) -> float:
        try:
            for keyword in keywords:
                pattern = rf'{keyword}[:\s(]*?(-?\d+(?:\.\d+)?)'
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return float(match.group(1))
            return default
        except Exception:
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
            reasoning="Gemini AI 配額已滿或不可用，使用原始分析",
            market_condition="未知",
            confidence_adjustment=0
        )
