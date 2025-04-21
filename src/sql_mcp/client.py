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

SYSTEM_MESSAGE = """You are a helpful assistant with access to various tools.
Your task is to assist users by providing accurate information from the database.
Always respond in a professional and friendly manner. Use tools provided.
Before executing any query, get details of the database by using the get_database_info tool, 
check for the table name and then execute the query.
When you don't know something or need information, say you don't know."""


llm = ChatGroq(model="deepseek-r1-distill-llama-70b")
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

                print("\nLangchain app...")

                try:
                    response = await agent.ainvoke(
                        {"messages": [HumanMessage(content="List out all the users in the database.")]}
                    )

                    if "messages" in response:
                        response["messages"] = ensure_tool_message_format(response["messages"])
                    print_formatted_response(response)
                    
                    
                except Exception as tool_exc:
                    print("Error occured:")
                    traceback.print_exception(
                        type(tool_exc), tool_exc, tool_exc.__traceback__
                    )

    except Exception as e:
        print(f"Error connecting to server: {e}")
        traceback.print_exception(type(e), e, e.__traceback__)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())