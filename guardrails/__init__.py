# Guardrails Module
# Protects the RAG agent from:
# - Prompt injection attacks
# - Toxic/harmful content
# - Off-topic questions
# - PII leakage
# - Hallucination indicators

from .input_guards import validate_input, InputValidationResult
from .output_guards import validate_output, OutputValidationResult
from .config import GuardrailConfig
