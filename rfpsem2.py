#streamlit web app that uses the semantic kernel to generate jokes
#on local machine run: streamlit run 001_jokewebapp.py

import base64
import logging
from uuid import uuid4
from pandas import DataFrame
import streamlit as st
import asyncio
from typing import Annotated
import os
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.utils.logging import setup_logging
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (AzureChatPromptExecutionSettings,)
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.kernel import Kernel
from semantic_kernel.functions import KernelArguments
from io import BytesIO
from semantic_kernel.agents.open_ai.azure_assistant_agent import AzureAssistantAgent
from semantic_kernel.agents.open_ai.open_ai_assistant_agent import OpenAIAssistantAgent
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.kernel import Kernel
from semantic_kernel.connectors.memory.azure_ai_search import AzureAISearchStore
from azure.core.credentials import AzureKeyCredential
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureTextCompletion, AzureTextEmbedding
from semantic_kernel.connectors.memory.azure_cognitive_search import AzureCognitiveSearchMemoryStore
from semantic_kernel.core_plugins import TextMemoryPlugin
from semantic_kernel.memory import SemanticTextMemory
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureTextEmbedding
from semantic_kernel.connectors.memory.azure_ai_search import AzureAISearchStore
from semantic_kernel.connectors.memory.azure_ai_search import AzureAISearchCollection

from service_settings import ServiceSettings
from plugins.rfprag import rfpchat
from plugins.historyrfp import historyrfpchat
from dataclasses import dataclass, field
from semantic_kernel.data import (
    DistanceFunction,
    IndexKind,
    VectorStoreRecordDataField,
    VectorStoreRecordDefinition,
    VectorStoreRecordKeyField,
    VectorStoreRecordVectorField,
    vectorstoremodel,
)
import pandas as pd
import semantic_kernel as sk

from dotenv import load_dotenv
import asyncio
from functools import reduce

load_dotenv()

st.set_page_config(layout="wide")

streaming = False

async def invoke_agent(agent: ChatCompletionAgent, input: str, chat: ChatHistory):
    """Invoke the agent with the user input."""
    chat.add_user_message(input)

    print(f"# {AuthorRole.USER}: '{input}'")
    print(f"User: {input}")
    print(f"chat: {chat}")

    if streaming:
        contents = []
        content_name = ""
        async for content in agent.invoke_stream(chat):
            content_name = content.name
            contents.append(content)
        streaming_chat_message = reduce(lambda first, second: first + second, contents)
        print(f"# {content.role} - {content_name or '*'}: '{streaming_chat_message}'")
        chat.add_message(streaming_chat_message)
    else:
        async for content in agent.invoke(chat):
            print(f"# {content.role} - {content.name or '*'}: '{content.content}'")
            chat.add_message(content)

COLLECTION_NAME = "vec"

# https://github.com/microsoft/semantic-kernel/blob/main/python/samples/getting_started_with_agents/step1_agent.py

async def agenticrfp():
    kernel = Kernel()

    service_id = "agent"

    # kernel.add_service(AzureChatCompletion(service_id=service_id))
    # Add Azure OpenAI chat completion
    chat_completion = AzureChatCompletion(
        deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )
    kernel.add_service(chat_completion)
    # Set the logging level for  semantic_kernel.kernel to DEBUG.
    setup_logging()
    logging.getLogger("kernel").setLevel(logging.DEBUG)

    # kernel.add_plugin(plugin=YourPlugin(), plugin_name="your_plugin")
    kernel.add_plugin(rfpchat(),plugin_name="RFP_Chat", description="This plugin provides the ability to chat about RFPs, PDF will be uploaded")
    kernel.add_plugin(historyrfpchat(),plugin_name="HistoryRFPContent_Chat", description="This plugin provides the ability to chat about historical RFPs")

    agent = ChatCompletionAgent(service_id=service_id, kernel=kernel, name="RFP Agent", instructions="You are AI Agent to provide knowledge about RFPs, based on the question pick the right agent to response. There are 1 one for rfp uploaded document and other for historical")

    # Define a thread and invoke the agent with the user input
    # thread_id = await agent.create_thread()

    chat = ChatHistory()

    # chat.add_user_message("<input>")

    if prompt := st.chat_input("Show me resources worked on water construction projects in the last 5 years"):
        #chat.add_user_message(prompt)
        #print(f"User: {prompt}")
        st.chat_message("user").markdown(prompt)
 
        response = await invoke_agent(agent, prompt, chat)
        # st.write(response)
        # latest_message = chat[-1]
        st.write(response)

    #async for content in agent.invoke(chat):
    #    chat.add_message(content)
    #    display_message(message)

if __name__ == "__main__":
    asyncio.run(agenticrfp())