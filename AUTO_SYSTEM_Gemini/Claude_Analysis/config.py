# config.py
"""
Central configuration for the AUTO_SYSTEM_Gemini project.
"""
import os

# --- API Key Configuration ---
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# --- Model Configuration ---
GEMINI_MODEL_NAME = "gemini-2.5-pro"
LLM_MODEL_NAME = "claude-3-5-sonnet-20241022"

# --- Directory Configuration ---
# Define all directory paths here to centralize project structure.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Directories for generated specifications and data
OUT_JSON_DIR = os.path.join(PROJECT_ROOT, "OUT_JSON")
UNITTEST_DATA_DIR = os.path.join(PROJECT_ROOT, "UNITTEST_DATA")
GENERATED_DATA_DIR = os.path.join(UNITTEST_DATA_DIR, "generated")

# Directories for code generation process
CODE_PROMPT_DIR = os.path.join(PROJECT_ROOT, "CODE_PROMPT")
TEMP_CODE_DIR = os.path.join(PROJECT_ROOT, "temp_code")
NO_ERROR_CODE_DIR = os.path.join(PROJECT_ROOT, "NO_ERROR_CODE")

# Directories for code concatenation
CONCATNATED_CODE_DIR = os.path.join(PROJECT_ROOT, "CONCATNATED_CODE")
PACKAGE_DIR = os.path.join(CONCATNATED_CODE_DIR, "PACKAGE")
HISTORY_DIR = os.path.join(CONCATNATED_CODE_DIR, "HISTORY")
DATA_MAKER_INIT_DIR = os.path.join(CONCATNATED_CODE_DIR, "data_maker_init")
