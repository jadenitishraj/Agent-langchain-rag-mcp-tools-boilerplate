"""
Input Guards - Validate and sanitize user input
Checks:
1. Prompt injection detection
2. Toxicity/harmful content
3. Topic relevance
4. Input length
"""
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional
from .config import GuardrailConfig, DEFAULT_CONFIG

@dataclass
class InputValidationResult:
    """Result of input validation."""
    is_valid: bool
    sanitized_input: str
    blocked_reason: Optional[str] = None
    warnings: List[str] = None
    scores: dict = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.scores is None:
            self.scores = {}

# ============================================================================
# PROMPT INJECTION DETECTION
# ============================================================================

INJECTION_PATTERNS = [
    # Direct instruction override
    r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|rules?)",
    r"disregard\s+(all\s+)?(previous|above|prior)",
    r"forget\s+(everything|all|what)\s+(you|i)\s+(said|told|wrote)",
    
    # Role manipulation
    r"you\s+are\s+now\s+(?:a|an|the)\s+\w+",
    r"pretend\s+(to\s+be|you\s+are)",
    r"act\s+as\s+(if|though|a|an)",
    r"roleplay\s+as",
    r"switch\s+(to|into)\s+\w+\s+mode",
    
    # System prompt extraction
    r"(show|tell|reveal|display|print|output)\s+(me\s+)?(your|the)\s+(system|initial|original)\s+(prompt|instructions?|message)",
    r"what\s+(are|is)\s+your\s+(instructions?|rules?|prompt)",
    r"repeat\s+(your|the)\s+(system|initial)\s+(prompt|message)",
    
    # Jailbreak attempts
    r"dan\s+mode",
    r"developer\s+mode",
    r"jailbreak",
    r"bypass\s+(the\s+)?(restrictions?|filters?|rules?)",
    r"override\s+(the\s+)?(safety|content)\s+(filters?|rules?)",
    
    # Delimiter injection
    r"```system",
    r"\[system\]",
    r"<\|im_start\|>",
    r"<\|endoftext\|>",
    
    # Code injection
    r"eval\s*\(",
    r"exec\s*\(",
    r"import\s+os",
    r"subprocess",
    r"__import__",
]

def detect_prompt_injection(text: str) -> Tuple[bool, float, List[str]]:
    """
    Detect prompt injection attempts.
    Returns: (is_injection, confidence, matched_patterns)
    """
    text_lower = text.lower()
    matched = []
    
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            matched.append(pattern)
    
    if matched:
        confidence = min(1.0, len(matched) * 0.3)  # More matches = higher confidence
        return True, confidence, matched
    
    return False, 0.0, []

# ============================================================================
# TOXICITY DETECTION
# ============================================================================

TOXIC_PATTERNS = [
    # Explicit hate speech
    r"\b(hate|kill|murder|attack)\s+(all\s+)?(jews?|muslims?|christians?|blacks?|whites?|asians?|gays?|women|men)\b",
    
    # Violence
    r"\b(how\s+to\s+)?(kill|murder|harm|hurt|attack|assault)\s+(someone|people|a\s+person)",
    r"\b(bomb|explosive|weapon)\s+(making|instructions?|how\s+to)",
    
    # Self-harm
    r"\b(how\s+to\s+)?(commit\s+)?suicide\b",
    r"\b(ways?\s+to\s+)?(hurt|harm|cut)\s+(myself|yourself)\b",
    
    # Illegal activities
    r"\b(how\s+to\s+)?(hack|steal|fraud|scam)\b",
    r"\b(drug|cocaine|heroin|meth)\s+(making|cooking|recipe)",
    
    # Explicit content
    r"\b(porn|xxx|nude|naked)\b",
]

TOXIC_WORDS = [
    "nigger", "faggot", "retard", "kike", "chink", "spic",
    "cunt", "bitch", "whore", "slut"  # Only when used as slurs
]

def detect_toxicity(text: str) -> Tuple[bool, float, List[str]]:
    """
    Detect toxic/harmful content.
    Returns: (is_toxic, confidence, reasons)
    """
    text_lower = text.lower()
    reasons = []
    
    # Check patterns
    for pattern in TOXIC_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            reasons.append(f"Pattern match: {pattern[:30]}...")
    
    # Check toxic words
    for word in TOXIC_WORDS:
        if word in text_lower:
            reasons.append(f"Toxic word: {word}")
    
    if reasons:
        confidence = min(1.0, len(reasons) * 0.4)
        return True, confidence, reasons
    
    return False, 0.0, []

# ============================================================================
# TOPIC RELEVANCE CHECK
# ============================================================================

def check_topic_relevance(text: str, allowed_topics: List[str]) -> Tuple[bool, float]:
    """
    Check if input is relevant to allowed topics.
    Returns: (is_relevant, relevance_score)
    """
    text_lower = text.lower()
    
    # Count topic keyword matches
    matches = 0
    for topic in allowed_topics:
        if topic in text_lower:
            matches += 1
    
    # Also check for general question words
    question_words = ["what", "how", "why", "who", "when", "where", "tell", "explain", "describe"]
    has_question = any(word in text_lower for word in question_words)
    
    # Calculate relevance
    if matches > 0:
        relevance = min(1.0, matches * 0.3)
        return True, relevance
    elif has_question:
        # Give benefit of doubt for questions
        return True, 0.3
    else:
        return False, 0.0

# ============================================================================
# INPUT LENGTH CHECK
# ============================================================================

def check_input_length(text: str, max_length: int) -> Tuple[bool, str]:
    """
    Check and truncate input if too long.
    Returns: (is_valid, truncated_text)
    """
    if len(text) <= max_length:
        return True, text
    
    # Truncate at word boundary
    truncated = text[:max_length].rsplit(' ', 1)[0]
    return False, truncated

# ============================================================================
# MAIN VALIDATION FUNCTION
# ============================================================================

def validate_input(
    text: str,
    config: GuardrailConfig = None
) -> InputValidationResult:
    """
    Validate user input through all guards.
    
    Args:
        text: The user's input text
        config: Guardrail configuration
    
    Returns:
        InputValidationResult with validation status and details
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    warnings = []
    scores = {}
    sanitized = text.strip()
    
    # 1. Length check
    if config.enable_input_length_check:
        is_valid_length, sanitized = check_input_length(sanitized, config.max_input_length)
        if not is_valid_length:
            warnings.append(f"Input truncated from {len(text)} to {len(sanitized)} characters")
    
    # 2. Prompt injection check
    if config.enable_injection_detection:
        is_injection, confidence, patterns = detect_prompt_injection(sanitized)
        scores["injection"] = confidence
        
        if is_injection and confidence > 0.5:
            return InputValidationResult(
                is_valid=False,
                sanitized_input=sanitized,
                blocked_reason="Potential prompt injection detected",
                warnings=warnings,
                scores=scores
            )
        elif is_injection:
            warnings.append("Low-confidence injection pattern detected")
    
    # 3. Toxicity check
    if config.enable_toxicity_check:
        is_toxic, confidence, reasons = detect_toxicity(sanitized)
        scores["toxicity"] = confidence
        
        if is_toxic and confidence >= config.toxicity_threshold:
            return InputValidationResult(
                is_valid=False,
                sanitized_input=sanitized,
                blocked_reason="Harmful or inappropriate content detected",
                warnings=warnings,
                scores=scores
            )
        elif is_toxic:
            warnings.append("Potentially sensitive content detected")
    
    # 4. Topic relevance check
    if config.enable_topic_check:
        is_relevant, relevance = check_topic_relevance(sanitized, config.allowed_topics)
        scores["relevance"] = relevance
        
        if not is_relevant:
            warnings.append("Question may be off-topic for this knowledge base")
    
    return InputValidationResult(
        is_valid=True,
        sanitized_input=sanitized,
        blocked_reason=None,
        warnings=warnings,
        scores=scores
    )
