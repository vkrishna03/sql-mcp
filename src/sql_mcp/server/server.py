import json
import logging
import uvicorn
import psycopg2
from psycopg2.extras import RealDictCursor

from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport

from starlette.requests import Request
from starlette.applications import Starlette
from starlette.routing import Route, Mount


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
def get_database_info() -> str:
    """Get information about database tables and their schemas"""
    logger.info("Fetching database structure information")
    
    conn = None
    try:
        conn = psycopg2.connect(
            DATABASE_URI,
            cursor_factory=RealDictCursor
        )
        # Set transaction to read-only
        conn.set_session(readonly=True)
        
        with conn.cursor() as cur:
            # Get tables
            cur.execute("""
                SELECT 
                    table_name,
                    (
                        SELECT json_agg(row_to_json(column_info))
                        FROM (
                            SELECT 
                                column_name,
                                data_type,
                                is_nullable,
                                column_default
                            FROM information_schema.columns c
                            WHERE c.table_name = t.table_name
                            AND c.table_schema = 'public'
                        ) column_info
                    ) as columns
                FROM information_schema.tables t
                WHERE table_schema = 'public'
                AND table_type = 'BASE TABLE'
            """)
            
            tables_info = cur.fetchall()
            return json.dumps(tables_info, indent=2, default=str)
            
    except Exception as e:
        logger.error(f"Database info error: {str(e)}")
        return json.dumps({"error": str(e)})
        
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

if __name__ == "__main__":
    uvicorn.run(starlette_app, host="localhost", port=8000)