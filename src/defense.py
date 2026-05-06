"""
Phase 3: Defense Layer - Jailbreak Detection
Simple keyword-based detection for prompt injection attempts
"""

import logging
from typing import Optional, List
from .models import JailbreakDetectionResult, ConfidenceLevel, AgentState, ConversationMessage

logger = logging.getLogger(__name__)


# Simple keyword patterns to detect jailbreak attempts
JAILBREAK_KEYWORDS = [
    "ignore", "forget", "override", "bypass", "admin", "developer",
    "system prompt", "instructions", "pretend", "roleplay", "act as",
    "disable safety", "without restrictions", "unfiltered", "raw output",
    "ignore guidelines", "no ethical", "no safety"
]


class JailbreakDetector:
    """Simple jailbreak detection"""
    
    def detect_jailbreak(self, text: str, conversation_history: Optional[List[ConversationMessage]] = None) -> JailbreakDetectionResult:
        
        logger.info("Running jailbreak detection")
        text_lower = text.lower()
        detected_keywords = []
        
        # Check for jailbreak keywords
        for keyword in JAILBREAK_KEYWORDS:
            if keyword in text_lower:
                detected_keywords.append(keyword)
        logger.info("Detected %d suspicious keywords", len(detected_keywords))

        # Determine if jailbreak based on keywords found
        is_jailbreak = len(detected_keywords) >= 1
        if is_jailbreak:
            logger.warning("Potential jailbreak attempt detected")
        # Set confidence level based on number of keywords
        if len(detected_keywords) >= 3:
            confidence = ConfidenceLevel.HIGH
        elif len(detected_keywords) >= 1:
            confidence = ConfidenceLevel.MEDIUM
        else:
            confidence = ConfidenceLevel.LOW
        
        reason = f"Found keywords: {', '.join(detected_keywords)}" if detected_keywords else None
        logger.info("Jailbreak detection completed")

        return JailbreakDetectionResult(
            is_jailbreak=is_jailbreak,
            confidence=confidence,
            patterns_detected=detected_keywords,
            reason=reason
        )


class DefenseEngine:
    """Handles jailbreak detection and blocking"""
    
    def __init__(self):
        self.detector = JailbreakDetector()
        logger.info("DefenseEngine initialized")
    
    def process_with_defense(self, state: AgentState) -> dict:
        """Check if query is jailbreak attempt"""
        logger.info("Processing query through defense layer")
        result = self.detector.detect_jailbreak(state.query)
        logger.info("Defense check completed. Blocked: %s", result.is_jailbreak)
        
        return {
            "is_blocked": result.is_jailbreak,
            "detection_result": result,
            "should_continue_processing": not result.is_jailbreak
        }
    
    def get_defense_response(self, bot_id: str) -> str:
        """Return bot's response to block jailbreak"""
        responses = {
            "bot_A": "I can't help with that request. I stick to ethical AI principles.",
            "bot_B": "That's a jailbreak attempt targeting privacy/ethics. Not engaging.",
            "bot_C": "This violates compliance protocols. I cannot process this request."
        }
        return responses.get(bot_id, "I cannot process this request.")
