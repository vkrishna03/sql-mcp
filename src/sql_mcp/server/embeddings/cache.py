import os
import json
import logging
import hashlib
import aiofiles
from sql_mcp.server.embeddings.config import CACHE_DIR

logger = logging.getLogger(__name__)

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_key(table_data):
    """Generate a deterministic cache key for a table schema."""
    table_name = table_data['table_name']
    # Create a hash of the column definitions to detect schema changes
    columns_str = json.dumps(table_data.get('columns', []), sort_keys=True)
    hash_obj = hashlib.md5(columns_str.encode())
    schema_hash = hash_obj.hexdigest()[:8]
    return f"{table_name}_{schema_hash}"

async def get_cached_description(cache_key):
    """Get a cached table description if available."""
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    try:
        if os.path.exists(cache_file):
            async with aiofiles.open(cache_file, 'r') as f:
                content = await f.read()
                return json.loads(content)
        return None
    except Exception as e:
        logger.error(f"Error reading cache file {cache_file}: {str(e)}")
        return None

async def save_to_cache(cache_key, description_data):
    """Save a table description to cache."""
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    try:
        async with aiofiles.open(cache_file, 'w') as f:
            await f.write(json.dumps(description_data))
        return True
    except Exception as e:
        logger.error(f"Error writing to cache file {cache_file}: {str(e)}")
        return False