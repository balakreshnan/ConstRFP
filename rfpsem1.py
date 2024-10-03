#streamlit web app that uses the semantic kernel to generate jokes
#on local machine run: streamlit run 001_jokewebapp.py

import base64
import streamlit as st
import asyncio
from typing import Annotated
import os
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
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
#from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential

from service_settings import ServiceSettings
from plugins.rfprag import rfpchat
from plugins.historyrfp import historyrfpchat
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(layout="wide")

@st.cache_resource
def setup_kernel_and_agent():
    kernel = Kernel()
    service_settings = ServiceSettings.create()
    # Remove all services so that this cell can be re-run without restarting the kernel
    kernel.remove_all_services()

    service_id = "default"
    kernel.add_service(AzureChatCompletion(service_id=service_id,),)
    kernel.add_plugin(rfpchat(),plugin_name="RFP_Chat",)
    kernel.add_plugin(historyrfpchat(),plugin_name="HistoryRFPContent_Chat",)
    chat_completion : AzureChatCompletion = kernel.get_service(type=ChatCompletionClientBase)

    return kernel, chat_completion

# Initialize kernel and function as global variables
global_kernel, global_chat_completion = setup_kernel_and_agent()


AGENT_NAME = "FileSearch"
AGENT_INSTRUCTIONS = "Find answers to the user's questions in the provided file."

# Note: you may toggle this to switch between AzureOpenAI and OpenAI
use_azure_openai = True

#vector_store = AzureAISearchStore()

# A helper method to invoke the agent with the user input
async def invoke_agent(agent: OpenAIAssistantAgent, thread_id: str, input: str) -> None:
    """Invoke the agent with the user input."""
    await agent.add_chat_message(thread_id=thread_id, message=ChatMessageContent(role=AuthorRole.USER, content=input))

    print(f"# {AuthorRole.USER}: '{input}'")

    async for content in agent.invoke(thread_id=thread_id):
        if content.role != AuthorRole.TOOL:
            print(f"# {content.role}: {content.content}")


async def rfpsemagent():
    # Create the instance of the Kernel
    kernel = Kernel()

    # Define a service_id for the sample
    service_id = "agent"

    # Get the path to the travelinfo.txt file
    pdf_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "resources", "Virginia Railway Express_2.pdf")

    #search_client = SearchIndexClient(endpoint="https://<your-search-service-name>.search.windows.net", credential="<your-search-service-key>")
    index_name = os.getenv("AZURE_AI_SEARCH_INDEX1")
    #search_client = SearchIndexClient(endpoint=os.getenv("AZURE_AI_SEARCH_ENDPOINT"), credential=os.getenv("AZURE_AI_SEARCH_KEY"), 
    #                                  index_name=index_name)
    search_client = SearchIndexClient(endpoint=os.getenv("AZURE_AI_SEARCH_ENDPOINT"), credential=AzureKeyCredential(os.getenv("AZURE_AI_SEARCH_KEY")))
    #vector_store = AzureAISearchStore(search_index_client=search_client)
    #store = AzureAISearchStore()
    store = AzureAISearchStore(search_index_client=search_client)
    assert store is not None
    

    # Create the agent configuration
    if use_azure_openai:
        agent = await AzureAssistantAgent.create(
            kernel=kernel,
            service_id=service_id,
            name=AGENT_NAME,
            instructions=AGENT_INSTRUCTIONS,
            enable_file_search=True,
            vector_store_filenames=[pdf_file_path],
        )
    else:
        agent = await OpenAIAssistantAgent.create(
            kernel=kernel,
            service_id=service_id,
            name=AGENT_NAME,
            instructions=AGENT_INSTRUCTIONS,
            enable_file_search=True,
            vector_store_filenames=[pdf_file_path],
        )

    # Define a thread and invoke the agent with the user input
    thread_id = await agent.create_thread()

    try:
        await invoke_agent(agent, thread_id=thread_id, input="Who is the youngest employee?")
        await invoke_agent(agent, thread_id=thread_id, input="Who works in sales?")
        await invoke_agent(agent, thread_id=thread_id, input="I have a customer request, who can help me?")
    finally:
        [await agent.delete_file(file_id) for file_id in agent.file_search_file_ids]
        await agent.delete_thread(thread_id)
        await agent.delete()

asyncio.run(rfpsemagent())  
#asyncio.run(rfpsem())  