from sql_mcp.server.embeddings.config import INDEX_NAME, METADATA_KEY
from sql_mcp.server.embeddings.vector_store import initialize_pinecone
from sql_mcp.server.embeddings.database import fetch_database_schema
from sql_mcp.server.embeddings.llm import textify_schema
from sql_mcp.server.embeddings.cache import get_cache_key

# Import the main functions to be exposed
from sql_mcp.server.embeddings.vector_store import search_schema
from sql_mcp.server.embeddings.core import (
    update_schema_embeddings,
    start_periodic_schema_updates,
    initialize_schema_embeddings
)

__all__ = [
    'search_schema',
    'update_schema_embeddings',
    'start_periodic_schema_updates',
    'initialize_schema_embeddings',
    'textify_schema'
]