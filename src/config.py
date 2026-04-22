import os
from dotenv import load_dotenv

load_dotenv()

# Routing thresholds
SIMILARITY_THRESHOLD = 0.65
FALLBACK_TO_BEST = True

# Search retry logic
SEARCH_MAX_RETRIES = 3
SEARCH_TIMEOUT = 5  # seconds

# Conversation history
MAX_CONVERSATION_HISTORY = 5

# LLM settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
LLM_TEMPERATURE = 0.7
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# DEFENSE CONFIGURATION 
JAILBREAK_CONFIDENCE_THRESHOLD = 0.85
ENABLE_DEFENSE_LOGGING = True

#  LOGGING 
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")