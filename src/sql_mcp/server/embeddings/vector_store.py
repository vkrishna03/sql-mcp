import logging
from pinecone import Pinecone, ServerlessSpec
from datetime import datetime, timedelta

from sql_mcp.server.embeddings.config import (
    PINECONE_API_KEY,
    INDEX_NAME,
    METADATA_KEY,
    PINECONE_CLOUD,
    PINECONE_REGION,
    SCHEMA_UPDATE_INTERVAL_DAYS
)
from sql_mcp.server.embeddings.llm import embedding_model

logger = logging.getLogger(__name__)

# Initialize Pinecone client and index
pc = None
index = None

def initialize_pinecone():
    """Initialize Pinecone client and index."""
    global pc, index
    
    if not PINECONE_API_KEY:
        logger.warning("PINECONE_API_KEY not set, embeddings storage disabled.")
        return None, None
    
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
        return pc, index
    except Exception as e:
        logger.error(f"Error initializing Pinecone: {str(e)}")
        return None, None

# Initialize Pinecone on module import
pc, index = initialize_pinecone()

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

def search_schema(query, top_k=5):
    """Search for schema information relevant to a query."""
    if not index:
        logger.warning("Pinecone index not available, cannot search schema")
        return None

    try:
        # Generate embedding for the query
        query_embedding = embedding_model.embed_query(query)

        # Search in Pinecone with the correct filter syntax
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter={"type": {"$eq": "table"}}  # Only match table vectors, not metadata
        )

        return results
    except Exception as e:
        logger.error(f"Error searching schema: {str(e)}")
        return None