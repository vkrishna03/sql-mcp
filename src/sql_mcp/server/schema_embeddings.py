import os
import json
import logging
import time
import threading
from datetime import datetime, timedelta
import uuid
import psycopg2
from pinecone import Pinecone, ServerlessSpec
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

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

pc = None
index = None

# Initialize Pinecone
if PINECONE_API_KEY:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if INDEX_NAME not in [idx['name'] for idx in pc.list_indexes()]:
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
else:
    logger.warning("PINECONE_API_KEY not set, embeddings storage disabled.")
    index = None

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

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

def textify_schema(table_data):
    """Convert table schema to a text string for embedding."""
    table_name = table_data['table_name']
    columns = table_data['columns'] if table_data['columns'] else []

    # Build column descriptions
    column_texts = []
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
        column_texts.append(col_text)

    # Combine into a single string
    return f"Table: {table_name}, Columns: {', '.join(column_texts)}"

def convert_to_embeddings(schema_data):
    """Convert schema JSON to embeddings and store in Pinecone."""
    if not schema_data or not index:
        logger.warning("Cannot convert to embeddings: missing schema data or Pinecone index")
        return False

    try:
        # First, delete existing schema embeddings to avoid stale data
        try:
            index.delete(filter={"$exists": "type"})
            logger.info("Cleared existing schema embeddings")
        except Exception as e:
            logger.warning(f"Error clearing existing schema embeddings: {str(e)}")

        vectors_to_upsert = []

        # Create vectors for table schemas
        for table in schema_data["tables"]:
            # Convert table schema to text
            schema_text = textify_schema(table)

            # Generate embedding
            embedding = embedding_model.embed_query(schema_text)

            # Create unique ID for the vector
            vector_id = f"table_{table['table_name']}"

            # Prepare metadata
            metadata = {
                'type': 'table',
                'table_name': table['table_name'],
                'schema_text': schema_text,
                'full_schema': json.dumps(table, default=str)
            }

            # Add to upsert list
            vectors_to_upsert.append({
                            'id': vector_id,
                            'values': embedding,
                            'metadata': metadata
                        })

        # Create a metadata vector with timestamp for tracking updates
        metadata_vector = {
            'id': METADATA_KEY,
            'values': embedding_model.embed_query(METADATA_KEY),  # Placeholder embedding
            'metadata': {
                'type': 'metadata',
                'last_updated': datetime.utcnow().isoformat(),
                'table_count': len(schema_data["tables"]),
                'relationship_count': len(schema_data["relationships"])
            }
        }
        vectors_to_upsert.append(metadata_vector)

        # Upsert to Pinecone
        index.upsert(vectors=vectors_to_upsert)
        logger.info(f"Upserted {len(vectors_to_upsert)} vectors to Pinecone index '{INDEX_NAME}'.")
        return True

    except Exception as e:
        logger.error(f"Error embedding schema: {str(e)}")
        return False

def should_update_schema():
    """Check if schema embeddings should be updated."""
    if not index:
        return False

    try:
        # Try to fetch the metadata vector
        results = index.fetch([METADATA_KEY])

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

def update_schema_embeddings():
    """Update schema embeddings if needed."""
    if not index:
        logger.warning("Pinecone index not available, skipping schema update")
        return False

    if should_update_schema():
        schema_data = fetch_database_schema()
        return convert_to_embeddings(schema_data)

    return False

def start_periodic_schema_updates():
    """Start a background thread to periodically update schema embeddings."""
    def update_thread():
        while True:
            try:
                update_schema_embeddings()
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
