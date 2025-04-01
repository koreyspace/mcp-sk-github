import os
import json
from dotenv import load_dotenv
from openai import AsyncOpenAI

import chainlit as cl
from mcp import ClientSession

from semantic_kernel.kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions import KernelFunction, kernel_function
from semantic_kernel.contents import ChatHistory, AuthorRole, ChatMessageContent
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.contents.function_call_content import FunctionCallContent
from semantic_kernel.contents.function_result_content import FunctionResultContent

# Load environment variables
load_dotenv()

# Example Weather Plugin (Tool)



class WeatherPlugin:
    @kernel_function(name="get_weather", description="Gets the weather for a city")
    def get_weather(self, city: str) -> str:
        """Retrieves the weather for a given city."""
        if "paris" in city.lower():
            return f"The weather in {city} is 20°C and sunny."
        elif "london" in city.lower():
            return f"The weather in {city} is 15°C and cloudy."
        else:
            return f"Sorry, I don't have the weather for {city}."





def flatten(xss):
    return [x for xs in xss for x in xs]


@cl.on_mcp_connect
async def on_mcp(connection, session: ClientSession):
    result = await session.list_tools()
    tools = [{
        "name": t.name,
        "description": t.description,
        "input_schema": t.inputSchema,
    } for t in result.tools]

    mcp_tools = cl.user_session.get("mcp_tools", {})
    mcp_tools[connection.name] = tools
    cl.user_session.set("mcp_tools", mcp_tools)


@cl.step(type="tool")
async def call_tool(tool_use):
    tool_name = tool_use.name
    tool_input = tool_use.input

    current_step = cl.context.current_step
    current_step.name = tool_name

    # Identify which mcp is used
    mcp_tools = cl.user_session.get("mcp_tools", {})
    mcp_name = None

    for connection_name, tools in mcp_tools.items():
        if any(tool.get("name") == tool_name for tool in tools):
            mcp_name = connection_name
            break

    if not mcp_name:
        current_step.output = json.dumps(
            {"error": f"Tool {tool_name} not found in any MCP connection"})
        return current_step.output

    mcp_session, _ = cl.context.session.mcp_sessions.get(mcp_name)

    if not mcp_session:
        current_step.output = json.dumps(
            {"error": f"MCP {mcp_name} not found in any MCP connection"})
        return current_step.output

    try:
        current_step.output = await mcp_session.call_tool(tool_name, tool_input)
    except Exception as e:
        current_step.output = json.dumps({"error": str(e)})

    return current_step.output


@cl.on_chat_start

async def on_chat_start():

        # Create AsyncOpenAI client with proper configuration
    client = AsyncOpenAI(
        api_key=os.getenv("GITHUB_TOKEN"),
        base_url="https://models.inference.ai.azure.com/"
    )

    # Create kernel
    kernel = Kernel()

    # Define service ID
    service_id = "agent"

    # Create and add chat completion service
    chat_completion_service = OpenAIChatCompletion(
        ai_model_id="gpt-4o",
        async_client=client,
        service_id=service_id
    )
    
    kernel.add_service(chat_completion_service)

    # Add plugins
    kernel.add_plugin(WeatherPlugin(), plugin_name="Weather")

    # Initialize Semantic Kernel

    # Create a new chat history
    chat_history = ChatHistory()

    # Store in user session
    cl.user_session.set("kernel", kernel)
    # Store the service in the session
    cl.user_session.set("chat_completion_service", chat_completion_service)

    cl.user_session.set("chat_history", chat_history)
    cl.user_session.set("mcp_tools", {})


@cl.on_message
async def on_message(message: cl.Message):
    # First get kernel from the user session
    kernel = cl.user_session.get("kernel")
    # Then use it to get settings
    settings = kernel.get_prompt_execution_settings_from_service_id(
        service_id="agent")
    
    chat_completion_service = cl.user_session.get("chat_completion_service")
    chat_history = cl.user_session.get("chat_history")

    # Add user message to history - use the simpler method
    chat_history.add_user_message(message.content)

    # Create a Chainlit message for the response stream
    answer = cl.Message(content="")

    async for msg in chat_completion_service.get_streaming_chat_message_content(
        chat_history=chat_history,
        user_input=message.content,
        settings=settings, 
        kernel=kernel,
    ):
        if msg.content:
            await answer.stream_token(msg.content)

    # Add the full assistant response to history
    chat_history.add_assistant_message(answer.content)

    # Send the final message
    await answer.send()
