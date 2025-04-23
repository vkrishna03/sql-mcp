import os
from dotenv import load_dotenv

load_dotenv()

# Database configuration
DATABASE_URI = os.getenv("DATABASE_URI", "postgresql://postgres:root@localhost:5432/enteam-main")

# Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = 'schema-embeddings'
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# Schema update configuration
SCHEMA_UPDATE_INTERVAL_DAYS = 7  # Update schema embeddings every week
METADATA_KEY = "schema_metadata"  # Key for storing metadata about the embeddings

# LLM configuration
MAX_CONCURRENT_LLM_CALLS = 1      # Maximum number of concurrent LLM calls
LLM_BACKOFF_FACTOR = 7            # Exponential backoff factor for rate limiting
MAX_RETRY_ATTEMPTS = 5            # Maximum number of retry attempts for API calls

# Cache configuration
CACHE_DIR = os.getenv("CACHE_DIR", ".cache")