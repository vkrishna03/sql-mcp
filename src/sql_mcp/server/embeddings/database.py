import json
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from sql_mcp.server.embeddings.config import DATABASE_URI

logger = logging.getLogger(__name__)

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