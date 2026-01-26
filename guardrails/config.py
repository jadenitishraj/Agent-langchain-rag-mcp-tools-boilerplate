"""
Guardrails Configuration
"""
from dataclasses import dataclass
from typing import List

@dataclass
class GuardrailConfig:
    """Configuration for all guardrails."""
    
    # Input Guards
    enable_injection_detection: bool = True
    enable_toxicity_check: bool = True
    enable_topic_check: bool = True
    enable_input_length_check: bool = True
    
    # Output Guards
    enable_pii_detection: bool = True
    enable_hallucination_check: bool = True
    enable_output_length_check: bool = True
    enable_relevance_check: bool = True
    
    # Thresholds
    max_input_length: int = 1000
    min_output_length: int = 50
    max_output_length: int = 3000
    toxicity_threshold: float = 0.7
    relevance_threshold: float = 0.3
    
    # Topic keywords (for codebase RAG)
    allowed_topics: List[str] = None
    
    def __post_init__(self):
        if self.allowed_topics is None:
            self.allowed_topics = [
                "langgraph", "langchain", "rag", "mcp", "agent",
                "guardrails", "fastapi", "react", "python", "javascript",
                "code", "function", "file", "implement", "explain",
                "architecture", "pipeline", "vector", "embedding", "tool"
            ]

# Default config
DEFAULT_CONFIG = GuardrailConfig()
