"""
Output Guards - Validate and sanitize LLM output
Checks:
1. PII detection & redaction
2. Hallucination indicators
3. Output length
4. Relevance to question
"""
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple
from .config import GuardrailConfig, DEFAULT_CONFIG

@dataclass
class OutputValidationResult:
    """Result of output validation."""
    is_valid: bool
    sanitized_output: str
    modified: bool = False
    warnings: List[str] = None
    scores: dict = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.scores is None:
            self.scores = {}

# ============================================================================
# PII DETECTION & REDACTION
# ============================================================================

PII_PATTERNS = {
    "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    "phone": r'\b(?:\+91[-.\s]?)?\d{10}\b|\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
    "credit_card": r'\b(?:\d{4}[-.\s]?){3}\d{4}\b',
    # Removed SSN and address patterns - too many false positives with URLs
}

# Whitelisted contact info (official/authorized info that should NOT be redacted)
WHITELISTED_CONTACT = [
    "support@agentforge.dev",
    "+1-555-123-4567",
    "5551234567",
]

def detect_and_redact_pii(text: str, whitelist: list = None) -> tuple:
    """
    Detect and redact PII from text, except whitelisted items.
    Returns: (redacted_text, pii_types_found, count)
    """
    if whitelist is None:
        whitelist = WHITELISTED_CONTACT
    
    redacted = text
    pii_found = []
    total_count = 0
    
    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, redacted, re.IGNORECASE)
        if matches:
            # Filter out whitelisted items
            matches_to_redact = [m for m in matches if m not in whitelist]
            
            if matches_to_redact:
                pii_found.append(pii_type)
                total_count += len(matches_to_redact)
                
                # Only redact non-whitelisted matches
                for match in matches_to_redact:
                    redacted = redacted.replace(match, f"[REDACTED_{pii_type.upper()}]")
    
    return redacted, pii_found, total_count

# ============================================================================
# HALLUCINATION DETECTION
# ============================================================================

HALLUCINATION_INDICATORS = [
    # Uncertainty phrases that might indicate hallucination
    r"i\s+(think|believe|assume|guess)\s+that",
    r"it\s+(?:might|may|could)\s+be\s+(?:that|possible)",
    r"i'm\s+not\s+(?:sure|certain|100%)",
    r"to\s+the\s+best\s+of\s+my\s+knowledge",
    r"if\s+i\s+recall\s+correctly",
    r"i\s+don't\s+have\s+(?:specific|exact)\s+information",
    
    # Making up specific facts
    r"according\s+to\s+(?:a\s+)?(?:study|research|survey)\s+(?:from|by|in)\s+\d{4}",
    r"statistics\s+show\s+that\s+\d+%",
    r"studies\s+have\s+shown\s+that",
    
    # Specific numbers without context
    r"exactly\s+\d+(?:,\d+)*\s+(?:people|users|cases)",
]

CONTRADICTION_PATTERNS = [
    r"however,\s+(?:on\s+the\s+other\s+hand|conversely|but)",
    r"while\s+this\s+is\s+true,\s+(?:it's\s+also|but)",
]

def detect_hallucination_risk(text: str, context: str = "") -> Tuple[float, List[str]]:
    """
    Detect potential hallucination indicators.
    Returns: (risk_score, indicators_found)
    """
    text_lower = text.lower()
    indicators = []
    
    for pattern in HALLUCINATION_INDICATORS:
        if re.search(pattern, text_lower):
            indicators.append(f"Pattern: {pattern[:40]}...")
    
    # Check for self-contradictions
    for pattern in CONTRADICTION_PATTERNS:
        if re.search(pattern, text_lower):
            indicators.append("Potential self-contradiction")
    
    # If context provided, check if response claims things not in context
    if context:
        # Simple check: if response mentions specific names/dates not in context
        context_words = set(re.findall(r'\b[A-Z][a-z]+\b', context))  # Proper nouns
        response_words = set(re.findall(r'\b[A-Z][a-z]+\b', text))
        new_proper_nouns = response_words - context_words - {"I", "The", "A", "An", "He", "She", "It", "They", "We", "You", "This", "That", "Python", "React", "FastAPI", "LangChain"}
        
        if len(new_proper_nouns) > 3:
            indicators.append(f"Introduced new proper nouns not in context: {list(new_proper_nouns)[:3]}")
    
    risk_score = min(1.0, len(indicators) * 0.2)
    return risk_score, indicators

# ============================================================================
# OUTPUT LENGTH CHECK
# ============================================================================

def check_output_length(
    text: str, 
    min_length: int, 
    max_length: int
) -> Tuple[bool, str, str]:
    """
    Check output length and adjust if needed.
    Returns: (is_valid, adjusted_text, issue_type)
    """
    if len(text) < min_length:
        return False, text, "too_short"
    
    if len(text) > max_length:
        # Truncate at sentence boundary
        truncated = text[:max_length]
        last_period = truncated.rfind('.')
        if last_period > max_length * 0.8:
            truncated = truncated[:last_period + 1]
        return False, truncated, "truncated"
    
    return True, text, "ok"

# ============================================================================
# RELEVANCE CHECK
# ============================================================================

def check_output_relevance(
    output: str, 
    question: str,
    threshold: float = 0.3
) -> Tuple[bool, float]:
    """
    Check if output is relevant to the question.
    Returns: (is_relevant, relevance_score)
    """
    question_lower = question.lower()
    output_lower = output.lower()
    
    # Extract key terms from question (excluding common words)
    stopwords = {"what", "how", "why", "who", "when", "where", "is", "are", "the", "a", "an", 
                 "does", "do", "did", "can", "could", "would", "should", "about", "tell", "me",
                 "please", "explain", "describe", "say", "says", "said", "your", "you", "i"}
    
    question_terms = set(re.findall(r'\b\w+\b', question_lower)) - stopwords
    question_terms = {t for t in question_terms if len(t) > 2}
    
    if not question_terms:
        return True, 1.0  # Can't check relevance
    
    # Count how many question terms appear in output
    matches = sum(1 for term in question_terms if term in output_lower)
    relevance = matches / len(question_terms) if question_terms else 1.0
    
    return relevance >= threshold, relevance

# ============================================================================
# MAIN VALIDATION FUNCTION
# ============================================================================

def validate_output(
    output: str,
    question: str = "",
    context: str = "",
    config: GuardrailConfig = None
) -> OutputValidationResult:
    """
    Validate LLM output through all guards.
    
    Args:
        output: The LLM's output text
        question: Original user question (for relevance check)
        context: RAG context (for hallucination check)
        config: Guardrail configuration
    
    Returns:
        OutputValidationResult with validation status and details
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    warnings = []
    scores = {}
    sanitized = output.strip()
    modified = False
    
    # 1. Length check
    if config.enable_output_length_check:
        is_valid_length, sanitized, issue = check_output_length(
            sanitized, 
            config.min_output_length, 
            config.max_output_length
        )
        
        if issue == "too_short":
            warnings.append("Response may be too brief")
        elif issue == "truncated":
            warnings.append(f"Response truncated to {len(sanitized)} characters")
            modified = True
    
    # 2. PII detection and redaction
    if config.enable_pii_detection:
        redacted, pii_types, pii_count = detect_and_redact_pii(sanitized)
        scores["pii_count"] = pii_count
        
        if pii_count > 0:
            sanitized = redacted
            modified = True
            warnings.append(f"Redacted {pii_count} PII instance(s): {', '.join(pii_types)}")
    
    # 3. Hallucination check
    if config.enable_hallucination_check:
        risk_score, indicators = detect_hallucination_risk(sanitized, context)
        scores["hallucination_risk"] = risk_score
        
        if risk_score > 0.5:
            warnings.append(f"High hallucination risk detected: {indicators[:2]}")
        elif risk_score > 0.2:
            warnings.append("Moderate hallucination indicators present")
    
    # 4. Relevance check
    if config.enable_relevance_check and question:
        is_relevant, relevance = check_output_relevance(sanitized, question, config.relevance_threshold)
        scores["relevance"] = relevance
        
        if not is_relevant:
            warnings.append(f"Response may not fully address the question (relevance: {relevance:.2f})")
    
    return OutputValidationResult(
        is_valid=True,  # We sanitize rather than block outputs
        sanitized_output=sanitized,
        modified=modified,
        warnings=warnings,
        scores=scores
    )
