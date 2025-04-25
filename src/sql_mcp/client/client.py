import asyncio
import traceback
import sys
import uuid

from dotenv import load_dotenv

from mcp.client.sse import sse_client
from mcp import ClientSession

from langchain import hub
from langgraph.prebuilt import create_react_agent
from langchain_groq.chat_models import ChatGroq
from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.tools import load_mcp_tools

load_dotenv()

SYSTEM_MESSAGE = """You are a helpful assistant that helps users interact with databases.
You have access to various tools that can help you understand the database structure and execute queries.

Follow these steps when handling user requests:
1. First use search_schema_for_query to get all relevant table names in the database.
2. Use get_table_structure to check the actual structure of relevant tables.
3. Only then craft and execute SQL queries for a postgres database based on verified table and column names.
Say if you got a table named user, you should query like SELECT * FROM public.user
4. Always execute queries with query_data_readonly to get results.
5. After getting results, format them nicely and explain them in natural language.

Be careful - the schema information from search_schema_for_query may be outdated or incorrect.
Always verify table structure before executing queries to avoid errors.
"""

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
# llm = ChatOllama(model="llama3.1")


def print_items(name: str, result: any) -> None:
    print(f"\nAvailable {name}")
    items = getattr(result, name)

    if items:
        for item in items:
            print(" ", item)
    else:
        print("No items available!")

def ensure_tool_message_format(messages):
    formatted_messages = []
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "tool":
            # Ensure content is a string
            if not isinstance(msg["content"], str):
                import json
                msg["content"] = json.dumps(msg["content"])
        formatted_messages.append(msg)
    return formatted_messages

def print_formatted_response(response):
    print("\n- - - -  Response of llm - - - - \n")
    for key, value in response.items():
        print(f"\nKey: {key}")
        if isinstance(value, list):
            for i, msg in enumerate(value):
                print(f"\nMessage {i + 1}:")
                print(f"Type: {type(msg).__name__}")
                print(f"Content: {msg.content}")
                if hasattr(msg, 'tool_calls') and msg.additional_kwargs.get('tool_calls'):
                    print("Tool Calls:")
                    for tool_call in msg.additional_kwargs['tool_calls']:
                        print(f"  Tool: {tool_call['function']['name']}")
                        print(f"  Arguments: {tool_call['function']['arguments']}")

async def process_query(agent, query):
    """Process a user query by letting the agent decide which tools to use"""
    print(f"\n--- Processing Query: {query} ---\n")

    try:
        response = await agent.ainvoke(
            {"messages": [HumanMessage(content=query)]}
        )

        if "messages" in response:
            response["messages"] = ensure_tool_message_format(response["messages"])
        print_formatted_response(response)

        return response
    except Exception as e:
        print("Error processing query:")
        traceback.print_exception(type(e), e, e.__traceback__)
        return None

async def interactive_mode(agent):
    """Run an interactive session"""
    print("\n=== Interactive Mode ===")
    print("Type your questions in natural language. Type 'exit' to quit.\n")

    while True:
        try:
            query = input("\nEnter your question: ")
            if query.lower() in ['exit', 'quit', 'q']:
                print("Exiting interactive mode.")
                break

            await process_query(agent, query)

        except KeyboardInterrupt:
            print("\nInterrupted. Exiting interactive mode.")
            break
        except Exception as e:
            print(f"Error: {str(e)}")


async def main():
    try:
        async with sse_client("http://localhost:8000/sse") as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                print("Connected to MCP server at ", "http://localhost:8000/sse")
                print_items("tools", await session.list_tools())
                print_items("resources", await session.list_resources())
                print_items("prompts", await session.list_prompts())

                tools = await load_mcp_tools(session)
                base_prompt = hub.pull("hwchase17/react")
                react_prompt_with_system = base_prompt.partial(system=SYSTEM_MESSAGE)

                agent = create_react_agent(model=llm, tools=tools, prompt=SYSTEM_MESSAGE)

                print("\nSQL Assistant...")

                # Example query
                await process_query(agent, "List out all the users,rewards and departments in the database.")

                # Interactive mode:
                # await interactive_mode(agent)
    except Exception as e:
        print(f"Error connecting to server: {e}")
        traceback.print_exception(type(e), e, e.__traceback__)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
