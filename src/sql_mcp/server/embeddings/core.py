import json
import logging
import time
import asyncio
import threading
from datetime import datetime

from sql_mcp.server.embeddings.config import METADATA_KEY
from sql_mcp.server.embeddings.database import fetch_database_schema
from sql_mcp.server.embeddings.llm import generate_enhanced_description, embedding_model
from sql_mcp.server.embeddings.vector_store import index, should_update_schema
from sql_mcp.server.utils.helpers import create_safe_event_loop, log_api_usage

logger = logging.getLogger(__name__)

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
            logger.info(f"Upserted {len(vectors_to_upsert)} vectors to Pinecone index.")
            return True
        else:
            logger.warning("No vectors to upsert after processing batch")
            return False

    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")
        return False

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
        loop, is_new_loop = create_safe_event_loop()
        
        try:
            result = loop.run_until_complete(update_schema_embeddings_async())
            log_api_usage("update_schema_embeddings", success=result)
            return result
        finally:
            if is_new_loop:
                loop.close()
    except Exception as e:
        logger.error(f"Error in update_schema_embeddings: {str(e)}")
        log_api_usage("update_schema_embeddings", success=False, error=e)
        return False

async def initialize_schema_embeddings():
    """Async-safe initialization for use in FastAPI/Starlette startup."""
    result = await update_schema_embeddings_async()
    if result:
        logger.info("Schema embeddings initialized successfully")
    else:
        logger.warning("Schema embeddings initialization not needed or failed")
    return result

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