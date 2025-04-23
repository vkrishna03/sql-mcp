import json
import asyncio
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def create_safe_event_loop():
    """Create a new event loop safely, handling existing loop scenarios."""
    try:
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're inside a running event loop, create a new one
                new_loop = asyncio.new_event_loop()
                return new_loop, True  # Return the loop and a flag indicating it's new
            else:
                # We have a loop but it's not running
                return loop, False
        except RuntimeError:
            # No event loop exists, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop, False  # It's a new loop but we've set it as the default
    except Exception as e:
        logger.error(f"Error creating event loop: {str(e)}")
        # Last resort fallback
        return asyncio.new_event_loop(), True

def format_table_data(table_data, format_type="string"):
    """Format table data into various formats.
    
    Args:
        table_data: The table data to format
        format_type: The output format ('string', 'dict', 'json')
        
    Returns:
        Formatted table representation
    """
    table_name = table_data.get('table_name', 'unknown')
    columns = table_data.get('columns', []) if isinstance(table_data, dict) else []
    
    if format_type == "dict":
        return {
            "table_name": table_name,
            "columns": [col['column_name'] for col in columns] if columns else [],
            "formatted_at": datetime.utcnow().isoformat()
        }
    elif format_type == "json":
        return json.dumps({
            "table_name": table_name,
            "columns": [col['column_name'] for col in columns] if columns else [],
            "formatted_at": datetime.utcnow().isoformat()
        }, indent=2)
    else:  # string format
        column_names = [col['column_name'] for col in columns] if columns else []
        return f"Table: {table_name}, Columns: {', '.join(column_names)}"

def sanitize_sql(sql):
    """Sanitize SQL queries to prevent injection attacks.
    
    Note: This is a basic implementation. For production, use prepared statements.
    """
    # Remove comments
    sanitized = ' '.join([line for line in sql.split('\n') 
                         if not line.strip().startswith('--')])
    
    # Check for dangerous operations
    dangerous_ops = ['DROP', 'TRUNCATE', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE']
    for op in dangerous_ops:
        if op in sanitized.upper().split():
            raise ValueError(f"SQL contains prohibited operation: {op}")
    
    return sanitized

def log_api_usage(function_name, success=True, error=None, metadata=None):
    """Log API usage for monitoring and analytics."""
    log_data = {
        "function": function_name,
        "timestamp": datetime.utcnow().isoformat(),
        "success": success
    }
    
    if error:
        log_data["error"] = str(error)
        
    if metadata:
        log_data["metadata"] = metadata
        
    logger.info(f"API USAGE: {json.dumps(log_data)}")