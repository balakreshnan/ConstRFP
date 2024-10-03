#streamlit web app that uses the semantic kernel to generate jokes
#on local machine run: streamlit run 001_jokewebapp.py

import base64
from uuid import uuid4
from pandas import DataFrame
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
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureTextCompletion, AzureTextEmbedding
from semantic_kernel.connectors.memory.azure_cognitive_search import AzureCognitiveSearchMemoryStore
from semantic_kernel.core_plugins import TextMemoryPlugin
from semantic_kernel.memory import SemanticTextMemory

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

from dotenv import load_dotenv

load_dotenv()

st.set_page_config(layout="wide")

COLLECTION_NAME = "vec"

@vectorstoremodel
@dataclass
class vec:
    chunk_id: Annotated[str, VectorStoreRecordKeyField()]
    title: Annotated[str, VectorStoreRecordDataField(is_full_text_searchable=True)]
    text_vector: Annotated[list[float], VectorStoreRecordVectorField(dimensions=1536, distance_function=DistanceFunction.COSINE, index_kind=IndexKind.HNSW)]
    chunk: Annotated[str, VectorStoreRecordDataField(is_filterable=True)]

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
from semantic_kernel.data import (
    VectorStoreRecordDataField,
    VectorStoreRecordDefinition,
    VectorStoreRecordKeyField,
    VectorStoreRecordVectorField,
)

vec_definition = VectorStoreRecordDefinition(
    fields={
        "chunk_id": VectorStoreRecordKeyField(property_type="str"),
        "title": VectorStoreRecordDataField(property_type="str", is_filterable=True),
        "chunk": VectorStoreRecordDataField(property_type="str", is_filterable=True),
        "text_vector": VectorStoreRecordVectorField(property_type="float", has_embedding=False, embedding_property_name="text_vector"),
    },
    container_mode=True,
    to_dict=lambda record, **_: record.to_dict(orient="records"),
    from_dict=lambda records, **_: DataFrame(records),
)

async def populate_memory(memory: SemanticTextMemory) -> None:
    # Add some documents to the ACS semantic memory
    await memory.save_information(COLLECTION_NAME, id="info1", text="My name is Andrea")
    await memory.save_information(COLLECTION_NAME, id="info2", text="I currently work as a tour guide")
    await memory.save_information(COLLECTION_NAME, id="info3", text="I've been living in Seattle since 2005")
    await memory.save_information(
        COLLECTION_NAME,
        id="info4",
        text="show me projects with railway construction in the last 5 years",
    )
    await memory.save_information(COLLECTION_NAME, id="info5", text="Show me construction projects in bridge")

async def search_acs_memory_questions(memory: SemanticTextMemory) -> None:
    questions = [
        "Summarize the content of the PDF file",
        "Summarize the content of the PDF file",
    ]

    for question in questions:
        print(f"Question: {question}")
        result = await memory.search(COLLECTION_NAME, question)
        print(f"Answer: {result[0].chunk}\n")

# A helper method to invoke the agent with the user input
async def invoke_agent(agent: OpenAIAssistantAgent, thread_id: str, input: str) -> None:
    """Invoke the agent with the user input."""
    returntxt = ""
    await agent.add_chat_message(thread_id=thread_id, message=ChatMessageContent(role=AuthorRole.USER, content=input))

    print(f"# {AuthorRole.USER}: '{input}'")

    async for content in agent.invoke(thread_id=thread_id):
        if content.role != AuthorRole.TOOL:
            #print(f"# {content.role}: {content.content}")
            returntxt = content.content
    return returntxt


async def rfpsemagent():
    # Create the instance of the Kernel
    kernel = Kernel()

    # Define a service_id for the sample
    service_id = "agent"
    vector_size = 1536

    # Get the path to the travelinfo.txt file
    pdf_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "resources", "Virginia Railway Express_2.pdf")

    #search_client = SearchIndexClient(endpoint="https://<your-search-service-name>.search.windows.net", credential="<your-search-service-key>")
    index_name = os.getenv("AZURE_AI_SEARCH_INDEX1")
    #search_client = SearchIndexClient(endpoint=os.getenv("AZURE_AI_SEARCH_ENDPOINT"), credential=os.getenv("AZURE_AI_SEARCH_KEY"), 
    #                                  index_name=index_name)
    # search_client = SearchIndexClient(endpoint=os.getenv("AZURE_AI_SEARCH_ENDPOINT"), credential=AzureKeyCredential(os.getenv("AZURE_AI_SEARCH_KEY")))
    #vector_store = AzureAISearchStore(search_index_client=search_client)
    store = AzureAISearchStore()
    #store = AzureAISearchStore(search_index_client=search_client)
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
    collection = store.get_collection(
        collection_name=COLLECTION_NAME, 
        data_model_type=pd.DataFrame, 
        data_model_definition=vec_definition,
    )

    # Define a thread and invoke the agent with the user input
    thread_id = await agent.create_thread()

    try:
        # Setting up OpenAI services for text completion and text embedding
        kernel.add_service(AzureTextCompletion(service_id="chunk"))
        async with AzureCognitiveSearchMemoryStore(vector_size=vector_size) as acs_connector:
            memory = SemanticTextMemory(storage=acs_connector, embeddings_generator=AzureTextEmbedding(service_id="text_vector"))
            kernel.add_plugin(TextMemoryPlugin(memory), "TextMemoryPlugin")
            # print("Populating memory...")
            # await populate_memory(memory)

            #print("Asking questions... (manually)")
            await search_acs_memory_questions(memory)

        # await invoke_agent(agent, thread_id=thread_id, input="Who is the youngest employee?")
        # await invoke_agent(agent, thread_id=thread_id, input="Who works in sales?")
        # await invoke_agent(agent, thread_id=thread_id, input="I have a customer request, who can help me?")
        # await invoke_agent(agent, thread_id=thread_id, input="Who is the oldest employee?")
        # await invoke_agent(agent, thread_id=thread_id, input="Summarize the PDF content?")
        if "history" not in st.session_state:  
            history = ChatHistory()
            st.session_state["history"] = history
            st.session_state["history"].add_system_message("You are a helpful assistant.") 

        for msg in st.session_state["history"]:
            print(msg.role + ":" + msg.content)
            if msg.role != AuthorRole.TOOL:
                with st.chat_message(msg.role):
                    st.markdown(msg.content)

        # React to user input
        if prompt := st.chat_input("Tell me about an email you want to send...(or something else)"):
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)
            result = await invoke_agent(agent, thread_id=thread_id, input=prompt)
            # await invoke_agent(agent, thread_id=thread_id, input="Summarize the PDF content?")
            print(result)
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(result)
    finally:
        [await agent.delete_file(file_id) for file_id in agent.file_search_file_ids]
        await agent.delete_thread(thread_id)
        await agent.delete()

asyncio.run(rfpsemagent())  
#asyncio.run(rfpsem())  