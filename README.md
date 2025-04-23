# SQL-MCP

**SQL-MCP** is an intelligent, open-source SQL database assistant that leverages Large Language Models (LLMs) and vector search to enable natural language interaction with your relational databases. It provides semantic schema search, safe query execution, and a conversational interface for both developers and non-technical users.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
- [Usage](#usage)
  - [Running the Server](#running-the-server)
  - [Running the Client](#running-the-client)
  - [Available Tools](#available-tools)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

---

## Features

- **Natural Language to SQL**: Ask questions in plain English and get SQL queries and results.
- **Semantic Schema Search**: Find relevant tables and columns using vector embeddings and LLMs.
- **Automatic Schema Embeddings**: Keeps vector representations of your schema up-to-date.
- **Safe Query Execution**: Only allows read-only queries, with SQL injection protection.
- **Interactive Chat Interface**: Use as a conversational assistant.
- **Extensible Tooling**: Easily add new tools and prompts.
- **Caching and Performance**: Caches LLM-generated schema descriptions for speed.

---

## Architecture

```
+-------------------+      +-------------------+      +-------------------+
|   Client (CLI)    | <--> |   MCP Server      | <--> |   Database        |
| (LangChain Agent) |      | (Starlette/FastAPI|      | (PostgreSQL)      |
+-------------------+      +-------------------+      +-------------------+
         |                          |
         |                          v
         |                +-------------------+
         |                |  Pinecone Vector  |
         |                |  Store (Schema)   |
         |                +-------------------+
         |                          |
         |                          v
         |                +-------------------+
         |                |   LLMs (Groq,     |
         |                |   HuggingFace)    |
         |                +-------------------+
```

- **Client**: Connects via SSE to the server, provides a conversational interface.
- **Server**: Exposes tools (search, query, schema info), manages schema embeddings, and orchestrates LLM calls.
- **Database**: PostgreSQL, schema introspected for embeddings.
- **Vector Store**: Pinecone, stores semantic representations of tables.
- **LLMs**: Used for schema description, query understanding, and agent reasoning.

---

## Project Structure

```
sql-mcp/
├── src/
│   └── sql_mcp/
│       ├── __init__.py
|       ├── client/
│       │   ├── __init__.py
│       │   ├── client.py            # CLI client with LLM agent
│       └── server/
│           ├── __init__.py
│           ├── server.py            # Main server (Starlette/FastAPI)
│           ├── tools.py             # Example MCP tools
│           ├── schema_embeddings.py # (Legacy/alt) embedding logic
│           ├── embeddings/
│           │   ├── __init__.py
│           │   ├── cache.py         # Caching for schema descriptions
│           │   ├── config.py        # All config/env
│           │   ├── core.py          # Embedding update logic
│           │   ├── database.py      # Schema introspection
│           │   ├── llm.py           # LLM integration for schema
│           │   └── vector_store.py  # Pinecone integration
│           └── utils/
│               ├── __init__.py
│               └── helpers.py       # Utility functions
├── .env.example                     # Example environment variables
├── pyproject.toml                   # Poetry/PEP 621 project config
├── README.md                        # This file
└── .gitignore
```

---

## Tech Stack

- **Python 3.11+**
- **PostgreSQL** (database)
- **Pinecone** (vector store)
- **LangChain** (LLM orchestration)
- **Groq** (LLM API, e.g. Llama-3/4)
- **HuggingFace** (embeddings)
- **Starlette** (server)
- **MCP** (Message Control Protocol, for agent/server communication)
- **aiofiles, backoff** (async, retries)

---

## Getting Started

### Prerequisites

- Python 3.11 or higher
- PostgreSQL database (local or remote)
- Pinecone account and API key
- Groq API key (for LLMs)
- (Optional) HuggingFace account for custom embeddings

### Installation

### 1. **Install Poetry**

If you don’t have Poetry installed, run:

```bash
pip install poetry
```
or follow the [official instructions](https://python-poetry.org/docs/#installation).

---

### 2. **Clone the Repository**

```bash
git clone https://github.com/your-org/sql-mcp.git
cd sql-mcp
```

---

### 3. **Set Up Environment Variables**

Copy the example environment file and edit it:

```bash
cp .env.example .env
```

Edit `.env` and fill in your credentials (database, Pinecone, Groq, etc).

---

### 4. **Install Dependencies**

Poetry will automatically create and use a virtual environment for the project:

```bash
poetry install
```

---

### 5. **Activate the Poetry Shell**

This ensures you’re using the correct virtual environment:

```bash
poetry env activate
source <path_above_cmd_generated>
```

You should now see your prompt prefixed with `(sql-mcp-...)`.

---

### 6. **Run the Server**

From the project root (where `pyproject.toml` is):

```bash
cd src
python -m sql_mcp.server.server
```

- The server will start at `http://localhost:8000`
- On startup, it will initialize schema embeddings and start a background thread for periodic updates.

---

### 7. **Run the Client**

Open a new terminal, activate the poetry shell again if needed:

```bash
cd sql-mcp
poetry shell
cd src
python -m sql_mcp.client
```

- The client connects to the server via SSE.
- You can run a single query or enter interactive mode (see `client.py`).

---

### Configuration

1. **Copy and edit the environment file**

```bash
cp .env.example .env
```

Edit `.env` and fill in your credentials:

```env
DATABASE_URI=postgresql://username:password@localhost:5432/your-db
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
GROQ_API_KEY=your-groq-api-key
CACHE_DIR=.cache
```

---

## Usage

### Running the Server

```bash
cd src
python -m sql_mcp.server.server
```

- The server will start at `http://localhost:8000`
- On startup, it will initialize schema embeddings and start a background thread for periodic updates.

### Running the Client

```bash
cd src
python -m sql_mcp.client
```

- The client connects to the server via SSE.
- You can run a single query or enter interactive mode (uncomment in `client.py`).

### Example Query

```python
await process_query(agent, "List out all the users and departments in the database.")
```

### Interactive Mode

Uncomment the line in `client.py`:

```python
# await interactive_mode(agent)
```

---

## Available Tools

The server exposes several tools to the agent:

- **say_hello**: Simple test tool.
- **query_data_readonly**: Execute read-only SQL queries (with SQL injection protection).
- **search_schema_for_query**: Semantic search for relevant tables given a natural language query.
- **get_table_structure**: Get detailed information about a table (columns, types, sample row).

You can add more tools in `server/tools.py`.

---

## Development

- **Schema Embeddings**: Managed in `server/embeddings/`. Embeddings are updated on startup and periodically.
- **LLM Integration**: Uses Groq for schema description and HuggingFace for embeddings.
- **Caching**: Table descriptions are cached in `.cache/` to avoid repeated LLM calls.
- **Safety**: All SQL queries are sanitized and executed in read-only mode.
- **Extensibility**: Add new tools/prompts using the MCP framework.

### Adding a New Tool

1. Edit `server/tools.py` or `server.py`
2. Use the `@mcp.tool()` decorator:

```python
@mcp.tool()
def my_tool(arg1: str) -> str:
    """Describe what this tool does."""
    # Your logic here
    return "result"
```

---

## Troubleshooting

- **Database Connection Issues**
  - Ensure PostgreSQL is running and `DATABASE_URI` is correct.
  - Check user/password/host/port.

- **Pinecone Issues**
  - Ensure your API key and region are correct.
  - Check Pinecone dashboard for index status.

- **LLM Issues**
  - Ensure your Groq API key is valid.
  - Monitor rate limits and quotas.

- **General**
  - Check logs for errors (`logging` is used throughout).
  - Use `DEBUG` log level for more details.



**Enjoy using SQL-MCP! If you have questions or suggestions, open an issue or pull request.**
