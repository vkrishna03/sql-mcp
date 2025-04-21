import logging
from typing import List, Dict, Any

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sql_mcp.server.server import mcp

logger = logging.getLogger(__name__)

DATABASE_URI="postgresql://postgres:root@localhost:5432/enteam-main"
engine = create_engine(DATABASE_URI)
Session = sessionmaker(bind=engine)


@mcp.tool()
def say_hello(text: str) -> str:
    return "Hello " + text; 

@mcp.tool()
def query_data(sql: str) -> str:
    """Execute SQL queries safely"""
    logger.info(f"Executing SQL query: {sql}")
    session = Session()
    try:
        result = session.execute(text(sql))
        rows = [dict(row._mapping) for row in result]
        session.commit()
        return rows
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        session.close()

@mcp.tool()
def execute_orm_query(table_name: str, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Execute queries using ORM style instead of raw SQL"""
    session = Session()
    try:
        # This is a simple example - you'd typically define proper SQLAlchemy models
        query = session.query(table_name)
        if filters:
            for key, value in filters.items():
                query = query.filter_by(**{key: value})
        return [dict(row._mapping) for row in query]
    except Exception as e:
        return [{"error": str(e)}]
    finally:
        session.close()

@mcp.prompt()
def example_prompt(code: str) -> str:
    return f"Please review this code:\n\n{code}"