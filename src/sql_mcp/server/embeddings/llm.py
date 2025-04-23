import re
import json
import logging
import asyncio
import backoff
from langchain_groq.chat_models import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from sql_mcp.server.embeddings.config import (
    MAX_CONCURRENT_LLM_CALLS,
    LLM_BACKOFF_FACTOR,
    MAX_RETRY_ATTEMPTS
)
from sql_mcp.server.embeddings.cache import (
    get_cache_key,
    get_cached_description,
    save_to_cache
)
from sql_mcp.server.embeddings.database import format_columns_for_llm
from sql_mcp.server.utils.helpers import create_safe_event_loop, format_table_data

logger = logging.getLogger(__name__)

# Initialize LLM and embedding model
embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
embedding_enhancement_llm = ChatGroq(model="llama3-70b-8192")

# Create a semaphore for limiting concurrent LLM calls
llm_semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)

@backoff.on_exception(
    backoff.expo,
    (Exception,),
    max_tries=MAX_RETRY_ATTEMPTS,
    factor=LLM_BACKOFF_FACTOR
)
async def call_llm_with_retries(messages):
    """Call LLM with exponential backoff for rate limiting."""
    async with llm_semaphore:
        return await embedding_enhancement_llm.ainvoke(messages)

async def generate_enhanced_description(table_data):
    """Generate an enhanced table description using LLM with rate limiting and caching."""
    table_name = table_data['table_name']
    columns = table_data['columns'] if table_data['columns'] else []

    # Generate cache key
    cache_key = get_cache_key(table_data)

    # Check cache first
    cached_data = await get_cached_description(cache_key)
    if cached_data:
        logger.info(f"Using cached description for table {table_name}")
        # Format the cached data into a string for embedding
        if isinstance(cached_data, dict):
            # Format the dictionary into the expected string format
            enhanced_text = f"""Table: {cached_data.get('table_name', table_name)}
Domain: {cached_data.get('domain', 'Unknown')}
Synonyms: {cached_data.get('synonyms', table_name)}
Description: {cached_data.get('description', f'Table storing {table_name} data')}
Columns: {', '.join(cached_data.get('columns', [col['column_name'] for col in columns]))}"""
            return enhanced_text
        return cached_data  # If it's already a string

    # Format column information for the LLM
    columns_text = format_columns_for_llm(columns)

    try:
        # Create prompt for the LLM
        system_prompt = """You are a database expert that helps create rich semantic descriptions of database tables.
Your task is to analyze a table's name and columns, then create:
1. A descriptive summary of what the table likely represents
2. A list of relevant synonyms and related terms for the table name
3. The semantic domain/category this table belongs to (e.g., user management, finance, etc.)

Your response should follow this format:
DESCRIPTION: [Clear description of what this table represents]
SYNONYMS: [Comma-separated list of synonyms and related terms]
DOMAIN: [The domain/category this table belongs to]"""

        user_prompt = f"""I need a rich semantic description for this database table:

Table Name: {table_name}

Columns:
{columns_text}

Please analyze this schema and provide a detailed description to help with semantic search."""

        # Call LLM with retries and rate limiting
        response = await call_llm_with_retries([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])

        response_text = response.content

        # Extract sections from response
        description_match = re.search(r'DESCRIPTION:(.*?)(?=SYNONYMS:|$)', response_text, re.DOTALL)
        synonyms_match = re.search(r'SYNONYMS:(.*?)(?=DOMAIN:|$)', response_text, re.DOTALL)
        domain_match = re.search(r'DOMAIN:(.*?)$', response_text, re.DOTALL)

        description = description_match.group(1).strip() if description_match else f"Table storing {table_name} data"
        synonyms = synonyms_match.group(1).strip() if synonyms_match else table_name
        domain = domain_match.group(1).strip() if domain_match else "Unknown"

        # Create enhanced description data
        description_data = {
            "table_name": table_name,
            "domain": domain,
            "synonyms": synonyms,
            "description": description,
            "columns": [col['column_name'] for col in columns],
            "generated_at": datetime.utcnow().isoformat()
        }

        # Save to cache
        await save_to_cache(cache_key, description_data)

        # Format for embedding
        enhanced_text = f"""Table: {table_name}
Domain: {domain}
Synonyms: {synonyms}
Description: {description}
Columns: {', '.join([col['column_name'] for col in columns])}"""

        logger.info(f"Created enhanced description for table {table_name}")
        return enhanced_text

    except Exception as e:
        logger.error(f"Error generating enhanced description for {table_name}: {str(e)}")

        # Create a basic fallback description
        fallback_text = f"""Table: {table_name}
Columns: {', '.join([col['column_name'] for col in columns])}"""

        return fallback_text

def textify_schema(table_data):
    """Synchronous wrapper for generate_enhanced_description."""
    try:
        import asyncio
        
        # Use our helper to get a safe event loop
        loop, is_new_loop = create_safe_event_loop()
        
        try:
            enhanced_description = loop.run_until_complete(generate_enhanced_description(table_data))
            
            # Ensure the result is a string
            if isinstance(enhanced_description, dict):
                return format_table_data(table_data, "string")
                
            return enhanced_description
        finally:
            # Close the loop if we created a new one
            if is_new_loop:
                loop.close()
                
    except Exception as e:
        logger.error(f"Error in textify_schema: {str(e)}")
        
        # Fall back to basic description
        return format_table_data(table_data, "string")

