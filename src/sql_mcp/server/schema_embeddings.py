import os
import re
import json
import logging
import time
import threading
import asyncio
from datetime import datetime, timedelta
import uuid
import hashlib
import psycopg2
from pinecone import Pinecone, ServerlessSpec
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_groq.chat_models import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import sqlalchemy as sa
from sqlalchemy import text
import aiofiles
import backoff
from concurrent.futures import ThreadPoolExecutor

# Setup logging
logger = logging.getLogger(__name__)

load_dotenv()

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
DATABASE_URI = os.getenv("DATABASE_URI", "postgresql://postgres:root@localhost:5432/enteam-main")
INDEX_NAME = 'schema-embeddings'
SCHEMA_UPDATE_INTERVAL_DAYS = 7  # Update schema embeddings every 7 days
METADATA_KEY = "schema_metadata"  # Key for storing metadata about the embeddings
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
CACHE_DIR = os.getenv("CACHE_DIR", ".cache")
MAX_CONCURRENT_LLM_CALLS = 1  # Maximum number of concurrent LLM calls
LLM_BACKOFF_FACTOR = 5  # Exponential backoff factor for rate limiting
MAX_RETRY_ATTEMPTS = 5  # Maximum number of retry attempts for API calls

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

pc = None
index = None
embedding_enhancement_llm = ChatGroq(model="llama3-70b-8192")

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Create a semaphore for limiting concurrent LLM calls
llm_semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)

# Initialize Pinecone
if PINECONE_API_KEY:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)

        # Check if index exists
        existing_indexes = [idx['name'] for idx in pc.list_indexes()]

        if INDEX_NAME not in existing_indexes:
            pc.create_index(
                name=INDEX_NAME,
                dimension=384,  # dimension for 'sentence-transformers/all-MiniLM-L6-v2'
                metric='cosine',
                spec=ServerlessSpec(
                    cloud=PINECONE_CLOUD,
                    region=PINECONE_REGION
                )
            )
            logger.info(f"Created new Pinecone index '{INDEX_NAME}'")

        index = pc.Index(INDEX_NAME)
        logger.info(f"Connected to Pinecone index '{INDEX_NAME}'")
    except Exception as e:
        logger.error(f"Error initializing Pinecone: {str(e)}")
        pc = None
        index = None
else:
    logger.warning("PINECONE_API_KEY not set, embeddings storage disabled.")
    index = None

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

def fetch_database_schema():
    """Fetch database schema information."""
    logger.info("Fetching database schema for embeddings")

    conn = None
    try:
        conn = psycopg2.connect(
            DATABASE_URI,
            cursor_factory=RealDictCursor
        )
        conn.set_session(readonly=True)

        with conn.cursor() as cur:
            # Get tables with columns
            cur.execute("""
                SELECT
                    t.table_name,
                    (
                        SELECT json_agg(row_to_json(column_info))
                        FROM (
                            SELECT
                                c.column_name,
                                c.data_type,
                                (c.is_nullable = 'YES') as is_nullable,
                                c.column_default,
                                (EXISTS (
                                    SELECT 1 FROM information_schema.table_constraints tc
                                    JOIN information_schema.constraint_column_usage ccu
                                    ON tc.constraint_name = ccu.constraint_name
                                    WHERE tc.constraint_type = 'PRIMARY KEY'
                                    AND tc.table_name = t.table_name
                                    AND ccu.column_name = c.column_name
                                )) as is_primary,
                                (EXISTS (
                                    SELECT 1 FROM information_schema.table_constraints tc
                                    JOIN information_schema.constraint_column_usage ccu
                                    ON tc.constraint_name = ccu.constraint_name
                                    JOIN information_schema.referential_constraints rc
                                    ON tc.constraint_name = rc.constraint_name
                                    WHERE tc.constraint_type = 'FOREIGN KEY'
                                    AND tc.table_name = t.table_name
                                    AND ccu.column_name = c.column_name
                                )) as is_foreign_key,
                                (
                                    SELECT ccu2.table_name
                                    FROM information_schema.table_constraints tc
                                    JOIN information_schema.constraint_column_usage ccu
                                    ON tc.constraint_name = ccu.constraint_name
                                    JOIN information_schema.referential_constraints rc
                                    ON tc.constraint_name = rc.constraint_name
                                    JOIN information_schema.constraint_column_usage ccu2
                                    ON rc.unique_constraint_name = ccu2.constraint_name
                                    WHERE tc.constraint_type = 'FOREIGN KEY'
                                    AND tc.table_name = t.table_name
                                    AND ccu.column_name = c.column_name
                                    LIMIT 1
                                ) as foreign_table
                            FROM information_schema.columns c
                            WHERE c.table_name = t.table_name
                            AND c.table_schema = 'public'
                        ) column_info
                    ) as columns
                FROM information_schema.tables t
                WHERE table_schema = 'public'
                AND table_type = 'BASE TABLE'
            """)

            tables = cur.fetchall()

            # Get table relationships
            cur.execute("""
                SELECT
                    tc.table_name as source_table,
                    kcu.column_name as source_column,
                    ccu.table_name as target_table,
                    ccu.column_name as target_column
                FROM
                    information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                    ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.constraint_column_usage ccu
                    ON tc.constraint_name = ccu.constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY'
            """)

            relationships = cur.fetchall()

            schema_data = {
                "tables": tables,
                "relationships": relationships
            }

            return schema_data

    except Exception as e:
        logger.error(f"Error fetching database schema: {str(e)}")
        return None

    finally:
        if conn:
            conn.rollback()
            conn.close()

def format_columns_for_llm(columns):
    """Format column information for LLM prompt."""
    column_descriptions = []
    for col in columns:
        col_name = col['column_name']
        data_type = col['data_type']
        constraints = []
        if col.get('is_primary', False):
            constraints.append('PRIMARY KEY')
        if col.get('is_nullable', True) == False:
            constraints.append('NOT NULL')
        if col.get('is_foreign_key', False):
            constraints.append(f"FOREIGN KEY to {col.get('foreign_table', 'unknown')}")
        col_text = f"{col_name} ({data_type}{', ' + ', '.join(constraints) if constraints else ''})"
        column_descriptions.append(col_text)

    return "\n".join([f"- {col}" for col in column_descriptions])

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
        system_prompt = """
            You are a database expert that helps create rich semantic descriptions of database tables.
            Your task is to analyze a table's name and columns, then create:
            1. A descriptive summary of what the table likely represents
            2. A list of relevant synonyms and related terms for the table name
            3. The semantic domain/category this table belongs to (e.g., user management, finance, etc.)

            Your response should follow this format:
            DESCRIPTION: [Clear description of what this table represents]
            SYNONYMS: [Comma-separated list of synonyms and related terms]
            DOMAIN: [The domain/category this table belongs to]
        """

        user_prompt = f"""
        I need a rich semantic description for this database table:

            Table Name: {table_name}

            Columns:
            {columns_text}

            Please analyze this schema and provide a detailed description to help with semantic search.
        """

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


async def process_table_batch(tables, relationships, is_batch=False):
    """Process a batch of tables asynchronously."""
    if not tables or not index:
        return False

    try:
        vectors_to_upsert = []

        # Process tables concurrently
        async def process_table(table):
            try:
                # Generate enhanced description
                schema_text = await generate_enhanced_description(table)

                # Ensure schema_text is a string
                if isinstance(schema_text, dict):
                    # Convert dict to string if needed
                    table_name = table['table_name']
                    columns = [col['column_name'] for col in table['columns']] if table['columns'] else []
                    schema_text = f"""Table: {table_name}
Columns: {', '.join(columns)}"""
                    logger.warning(f"Converted dict schema description to string for table {table_name}")

                # Generate embedding
                embedding = embedding_model.embed_query(schema_text)

                return {
                    'id': f"table_{table['table_name']}",
                    'values': embedding,
                    'metadata': {
                        'type': 'table',
                        'table_name': table['table_name'],
                        'schema_text': schema_text,
                        'full_schema': json.dumps(table, default=str)
                    }
                }
            except Exception as e:
                logger.error(f"Error processing table {table['table_name']}: {str(e)}")
                return None

        # Process tables with concurrency control
        tasks = []
        for table in tables:
            tasks.append(process_table(table))

        results = await asyncio.gather(*tasks)

        # Filter out None results and add to vectors
        vectors_to_upsert = [r for r in results if r is not None]

        # Add metadata vector on first batch
        if not is_batch:
            # Create a metadata vector with timestamp for tracking updates
            metadata_vector = {
                'id': METADATA_KEY,
                'values': embedding_model.embed_query(METADATA_KEY),  # Placeholder embedding
                'metadata': {
                    'type': 'metadata',
                    'last_updated': datetime.utcnow().isoformat(),
                    'table_count': len(tables),
                    'relationship_count': len(relationships) if relationships else 0
                }
            }
            vectors_to_upsert.append(metadata_vector)

        # Upsert to Pinecone
        if vectors_to_upsert:
            index.upsert(vectors=vectors_to_upsert)
            logger.info(f"Upserted {len(vectors_to_upsert)} vectors to Pinecone index '{INDEX_NAME}'.")
            return True
        else:
            logger.warning("No vectors to upsert after processing batch")
            return False

    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")
        return False

def textify_schema(table_data):
    """Synchronous wrapper for generate_enhanced_description."""
    try:
        import asyncio

        # Try to get an event loop, create one if there isn't one
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        enhanced_description = loop.run_until_complete(generate_enhanced_description(table_data))

        # Ensure the result is a string
        if isinstance(enhanced_description, dict):
            table_name = table_data['table_name']
            columns = table_data['columns'] if table_data['columns'] else []
            column_names = [col['column_name'] for col in columns]
            return f"Table: {table_name}, Columns: {', '.join(column_names)}"

        return enhanced_description
    except Exception as e:
        logger.error(f"Error in textify_schema: {str(e)}")

        # Fall back to basic description
        table_name = table_data['table_name']
        columns = table_data['columns'] if table_data['columns'] else []

        column_names = [col['column_name'] for col in columns]
        return f"Table: {table_name}, Columns: {', '.join(column_names)}"

def should_update_schema():
    """Check if schema embeddings should be updated."""
    if not index:
        return False

    try:
        # Try to fetch the metadata vector
        results = index.fetch(ids=[METADATA_KEY])

        if METADATA_KEY not in results['vectors']:
            logger.info("No schema metadata found, will create initial embeddings.")
            return True

        metadata = results['vectors'][METADATA_KEY]['metadata']
        last_updated = datetime.fromisoformat(metadata['last_updated'])

        # Check if it's time to update
        update_threshold = datetime.utcnow() - timedelta(days=SCHEMA_UPDATE_INTERVAL_DAYS)

        if last_updated < update_threshold:
            logger.info(f"Schema embeddings are outdated (last updated: {last_updated.isoformat()}), will update.")
            return True
        else:
            logger.info(f"Schema embeddings are current (last updated: {last_updated.isoformat()}), no update needed.")
            return False

    except Exception as e:
        logger.error(f"Error checking schema metadata: {str(e)}")
        return True  # Default to updating on error

async def initialize_schema_embeddings():
    """Async-safe initialization for use in FastAPI/Starlette startup."""
    result = await update_schema_embeddings_async()
    if result:
        logger.info("Schema embeddings initialized successfully")
    else:
        logger.warning("Schema embeddings initialization not needed or failed")
    return result


async def update_schema_embeddings_async():
    """Update schema embeddings asynchronously."""
    if not index:
        logger.warning("Pinecone index not available, skipping schema update")
        return False

    if not should_update_schema():
        return False

    schema_data = fetch_database_schema()

    if not schema_data:
        return False

    # Process tables in batches to avoid timeouts
    batch_size = 5  # Small batch size for better parallelism
    tables = schema_data["tables"]
    relationships = schema_data["relationships"]
    total_tables = len(tables)

    logger.info(f"Processing {total_tables} tables for embeddings in batches of {batch_size}")

    # Clear existing embeddings before first batch
    try:
        # Get all existing vector IDs
        # First, let's get all vectors with metadata.type = "table"
        try:
            # For newer Pinecone versions, use the proper filter syntax
            table_query_result = index.query(
                vector=[0.0] * 384,  # Dummy vector for query
                top_k=10000,  # Get a large number to catch all
                include_metadata=False,
                filter={"type": {"$eq": "table"}}
            )

            # Extract IDs
            table_ids = [match['id'] for match in table_query_result['matches']]

            # Also get the metadata key
            metadata_ids = [METADATA_KEY]

            # Combine IDs
            all_ids = table_ids + metadata_ids

            # Delete by IDs if we found any
            if all_ids:
                index.delete(ids=all_ids)
                logger.info(f"Cleared {len(all_ids)} existing schema embeddings")
            else:
                logger.info("No existing schema embeddings to clear")

        except Exception as filter_error:
            logger.warning(f"Error using filter query, falling back to list-delete approach: {str(filter_error)}")

            # Fallback: List all vectors and delete
            try:
                # Get all vector IDs (up to a reasonable limit)
                all_vectors = index.fetch(ids=[], limit=10000)
                all_ids = list(all_vectors['vectors'].keys())

                if all_ids:
                    index.delete(ids=all_ids)
                    logger.info(f"Cleared {len(all_ids)} existing schema embeddings using list-delete approach")
                else:
                    logger.info("No existing schema embeddings to clear")
            except Exception as list_error:
                logger.error(f"Error with list-delete fallback: {str(list_error)}")
                # Continue anyway to add new embeddings

    except Exception as e:
        logger.warning(f"Error clearing existing schema embeddings: {str(e)}")

    # Process batches
    success = True
    for i in range(0, total_tables, batch_size):
        batch = tables[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(total_tables + batch_size - 1)//batch_size}")

        # Process batch
        batch_success = await process_table_batch(batch, relationships, is_batch=(i > 0))

        if not batch_success:
            logger.error(f"Failed to process batch {i//batch_size + 1}")
            success = False

        # Add delay between batches
        await asyncio.sleep(1)

    return success

def update_schema_embeddings():
    """Synchronous wrapper for update_schema_embeddings_async."""
    try:
        import asyncio

        # Try to get an event loop, create one if there isn't one
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                logger.warning("Called update_schema_embeddings from within a running event loop")
                # Create a new loop for this task to avoid nested event loop issues
                new_loop = asyncio.new_event_loop()
                result = new_loop.run_until_complete(update_schema_embeddings_async())
                new_loop.close()
                return result
            else:
                # Use the existing loop
                return loop.run_until_complete(update_schema_embeddings_async())
        except RuntimeError:
            # No event loop exists, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(update_schema_embeddings_async())
            loop.close()
            return result
    except Exception as e:
        logger.error(f"Error in update_schema_embeddings: {str(e)}")
        return False

def start_periodic_schema_updates():
    """Start a background thread to periodically update schema embeddings."""
    def update_thread():
        while True:
            try:
                result = update_schema_embeddings()
                if result:
                    logger.info("Successfully updated schema embeddings")
                else:
                    logger.info("No schema embedding update was needed or update failed")
            except Exception as e:
                logger.error(f"Error in schema update thread: {str(e)}")

            # Sleep for a day before checking again
            time.sleep(86400)  # 24 hours in seconds

    thread = threading.Thread(target=update_thread, daemon=True)
    thread.start()
    logger.info("Started background thread for periodic schema updates")

def search_schema(query, top_k=5):
    """Search for schema information relevant to a query."""
    if not index:
        logger.warning("Pinecone index not available, cannot search schema")
        return None

    try:
        # Generate embedding for the query
        query_embedding = embedding_model.embed_query(query)

        # Search in Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter={"type": "table"}  # Only match table vectors, not metadata
        )

        return results
    except Exception as e:
        logger.error(f"Error searching schema: {str(e)}")
        return None
