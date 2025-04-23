import os
import json
import logging
import uvicorn
import psycopg2
import pinecone
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport

from starlette.requests import Request
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from sql_mcp.server.schema_embeddings import (
    update_schema_embeddings,
    start_periodic_schema_updates,
    search_schema
)


mcp = FastMCP("sql-mcp-server")
sse = SseServerTransport("/messages/")

logger = logging.getLogger(__name__)

## DB Config
DATABASE_URI="postgresql://postgres:root@localhost:5432/enteam-main"

@mcp.tool()
def say_hello(text: str) -> str:
    return "Hello " + text;


@mcp.tool()
def query_data_readonly(sql: str) -> str:
    """Execute read-only SQL queries and return results as JSON"""
    logger.info(f"Executing read-only SQL query: {sql}")

    conn = None
    try:
        conn = psycopg2.connect(
            DATABASE_URI,
            cursor_factory=RealDictCursor
        )
        # Set transaction to read-only
        conn.set_session(readonly=True)

        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            # RealDictCursor returns results as dictionaries
            return json.dumps(rows, default=str)

    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        return json.dumps({"error": str(e)})

    finally:
        if conn:
            conn.rollback()  # Always rollback read-only transaction
            conn.close()

@mcp.tool()
def search_schema_for_query(query: str) -> str:
    """Search the database schema for relevant tables based on a natural language query."""

    # Log the original query
    logger.info(f"Searching schema for: {query}")

    # Add some common variations to improve search
    expanded_query = f"{query} table schema columns users user accounts members customers profiles"

    # Perform the search with expanded query and lower threshold
    results = search_schema(expanded_query, top_k=10)

    if not results or not results['matches']:
        return json.dumps({
            "message": "No relevant schema information found",
            "success": False
        })

    relevant_tables = []
    for match in results['matches']:
        if match['score'] < 0.4:  # Threshold for relevance
            continue

        metadata = match['metadata']
        if metadata.get('type') == 'table':
            table_info = {
                'table_name': metadata['table_name'],
                'schema_text': metadata['schema_text'],
                'relevance_score': match['score'],
            }
            relevant_tables.append(table_info)

    return json.dumps({
        "message": f"Found {len(relevant_tables)} relevant tables",
        "relevant_tables": relevant_tables,
        "success": True
    }, indent=2)

@mcp.tool()
def list_all_tables() -> str:
    """List all tables in the database"""
    logger.info("Listing all tables in database")

    conn = None
    try:
        conn = psycopg2.connect(
            DATABASE_URI,
            cursor_factory=RealDictCursor
        )
        conn.set_session(readonly=True)

        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_type = 'BASE TABLE'
                ORDER BY table_name;
            """)

            tables = [row['table_name'] for row in cur.fetchall()]

            # Group tables by potential domain
            user_related = []
            department_related = []
            other_tables = []

            keywords = {
                'user': ['user', 'account', 'person', 'profile', 'member', 'customer', 'login', 'auth'],
                'department': ['department', 'dept', 'team', 'group', 'division', 'unit', 'org']
            }

            for table in tables:
                table_lower = table.lower()
                if any(keyword in table_lower for keyword in keywords['user']):
                    user_related.append(table)
                elif any(keyword in table_lower for keyword in keywords['department']):
                    department_related.append(table)
                else:
                    other_tables.append(table)

            return json.dumps({
                "total_tables": len(tables),
                "user_related_tables": user_related,
                "department_related_tables": department_related,
                "other_tables": other_tables,
                "all_tables": tables,
                "success": True
            }, indent=2)

    except Exception as e:
        logger.error(f"Error listing tables: {str(e)}")
        return json.dumps({
            "error": str(e),
            "success": False
        })

    finally:
        if conn:
            conn.rollback()
            conn.close()

@mcp.tool()
def get_table_structure(table_name: str) -> str:
    """Get detailed information about a specific table structure"""
    logger.info(f"Fetching structure for table: {table_name}")

    conn = None
    try:
        conn = psycopg2.connect(
            DATABASE_URI,
            cursor_factory=RealDictCursor
        )
        conn.set_session(readonly=True)

        with conn.cursor() as cur:
            # First check if the table exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = %s
                );
            """, (table_name,))

            if not cur.fetchone()['exists']:
                return json.dumps({
                    "error": f"Table '{table_name}' does not exist in the database.",
                    "success": False
                })

            # Get column information
            cur.execute("""
                SELECT
                    column_name,
                    data_type,
                    is_nullable,
                    column_default,
                    (
                        SELECT constraint_type
                        FROM information_schema.table_constraints tc
                        JOIN information_schema.constraint_column_usage ccu
                            ON tc.constraint_name = ccu.constraint_name
                        WHERE tc.table_name = c.table_name
                        AND ccu.column_name = c.column_name
                        AND tc.constraint_type = 'PRIMARY KEY'
                        LIMIT 1
                    ) as primary_key
                FROM information_schema.columns c
                WHERE table_name = %s
                AND table_schema = 'public'
                ORDER BY ordinal_position;
            """, (table_name,))

            columns = cur.fetchall()

            # Get a sample row if available
            try:
                cur.execute(f"SELECT * FROM \"{table_name}\" LIMIT 1")
                sample = cur.fetchone()
            except:
                sample = None

            return json.dumps({
                "table_name": table_name,
                "columns": columns,
                "sample_row": sample,
                "success": True
            }, default=str, indent=2)

    except Exception as e:
        logger.error(f"Error fetching table structure: {str(e)}")
        return json.dumps({
            "error": str(e),
            "success": False
        })

    finally:
        if conn:
            conn.rollback()
            conn.close()



async def handle_sse(request: Request) -> None:
    _server = mcp._mcp_server
    async with sse.connect_sse(request.scope, request.receive, request._send) as (reader, writer):
        await _server.run(reader, writer, _server.create_initialization_options())


starlette_app = Starlette(
    debug=True,
    routes=[
        Route("/sse", endpoint=handle_sse),
        Mount("/messages/", sse.handle_post_message)
    ]
)

@starlette_app.on_event("startup")
def startup_event():
    # Update schema embeddings synchronously at startup
    update_schema_embeddings()
    # Start background thread for periodic updates
    start_periodic_schema_updates()
    logger.info("Schema embeddings initialized")

if __name__ == "__main__":
    uvicorn.run(starlette_app, host="localhost", port=8000)
