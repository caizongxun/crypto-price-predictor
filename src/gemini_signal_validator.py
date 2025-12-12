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
            self.api_key = api_key
            # 優先使用 1.5-flash (配額較多: 15 RPM)
            # 如果失敗會自動降級
            self.model_name = 'gemini-1.5-flash'
            self.model = self._init_model(self.model_name)
        except Exception as e:
            logger.error(f"❌ Gemini 連接失敗: {e}")
            self.model = None

    def _init_model(self, model_name):
        try:
            model = genai.GenerativeModel(model_name)
            logger.info(f"✅ Gemini 模型已連接: {model_name}")
            return model
        except Exception as e:
            logger.warning(f"⚠️ 模型 {model_name} 初始化失敗: {e}")
            return None
    
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
            # 嘗試重新連接
            self.model = self._init_model('gemini-1.5-flash')
            if not self.model:
                return self._create_default_analysis(signal_type, confidence)
        
        try:
            # 智能延遲: 避免 429 錯誤
            # 1.5-flash 允許 15 RPM (每4秒一請求)
            # 2.0-flash-exp 允許 10 RPM (每6秒一請求)
            delay = 4.0 if '1.5' in self.model_name else 6.0
            time.sleep(delay)
            
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
            
            return self._generate_with_retry(prompt, symbol, signal_type, confidence, current_price)
        
        except Exception as e:
            logger.error(f"❌ Gemini 分析失敗 ({symbol}): {e}")
            return self._create_default_analysis(signal_type, confidence)

    def _generate_with_retry(self, prompt, symbol, signal_type, confidence, current_price, retry_count=0):
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
            return self._parse_gemini_response(response.text, signal_type, confidence, current_price)
            
        except Exception as e:
            error_msg = str(e)
            
            # 處理 404 (模型未找到) -> 自動切換模型
            if "404" in error_msg and retry_count < 2:
                logger.warning(f"⚠️ 模型 {self.model_name} 未找到，嘗試切換...")
                # 如果當前是 1.5，切換到 2.0-flash-exp
                # 如果當前是 2.0，切換到 gemini-pro
                if '1.5' in self.model_name:
                    self.model_name = 'gemini-2.0-flash-exp'
                else:
                    self.model_name = 'gemini-pro'
                
                self.model = self._init_model(self.model_name)
                if self.model:
                    time.sleep(2)
                    return self._generate_with_retry(prompt, symbol, signal_type, confidence, current_price, retry_count + 1)

            # 處理 429 (配額滿) -> 等待後重試
            if ("429" in error_msg or "Quota" in error_msg) and retry_count < 1:
                wait_time = 15  # 增加等待時間
                logger.warning(f"⚠️ 配額滿 ({symbol})，等待 {wait_time} 秒後重試...")
                time.sleep(wait_time)
                return self._generate_with_retry(prompt, symbol, signal_type, confidence, current_price, retry_count + 1)
            
            logger.error(f"❌ Gemini 請求最終失敗: {error_msg}")
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
