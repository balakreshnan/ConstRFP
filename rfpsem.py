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
from azure.search.documents.indexes import SearchIndexClient
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

# A helper method to invoke the agent with the user input
async def invoke_agent(input_text):
       
    # Enable planning
    execution_settings = AzureChatPromptExecutionSettings(tool_choice="auto")
    execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    st.session_state["history"].add_user_message(input_text)

    result = (await global_chat_completion.get_chat_message_contents(
            chat_history=st.session_state["history"],
            settings=execution_settings,
            kernel=global_kernel,
            arguments=KernelArguments(),
        ))[0]
    print(str(result))

    st.session_state["history"].add_assistant_message(str(result))
    return str(result)

AGENT_NAME = "FileSearch"
AGENT_INSTRUCTIONS = "Find answers to the user's questions in the provided file."

# Note: you may toggle this to switch between AzureOpenAI and OpenAI
use_azure_openai = True

#vector_store = AzureAISearchStore()

async def rfpsem():

    #st.sidebar.title("Virtuoso - Customer Account Planning Assistant")
    #st.sidebar.image("https://i.imgur.com/jxSzGbM.jpg", use_column_width=True)
    st.write("## Microsoft Construction Copilot")
    count = 0
    temp_file_path = ""
    pdf_bytes = None
    rfpcontent = {}
    rfplist = []
    #tab1, tab2, tab3, tab4 = st.tabs('RFP PDF', 'RFP Research', 'Draft', 'Create Word')
    modeloptions1 = ["gpt-4o-2", "gpt-4o-g", "gpt-4o", "gpt-4-turbo", "gpt-35-turbo"]

    # Create a dropdown menu using selectbox method
    selected_optionmodel1 = st.selectbox("Select an Model:", modeloptions1)
    count += 1

    col1, col2 = st.columns([1, 2])

    with col1:
        st.title("Microsoft Construction Copilot")
        reset = st.button('Reset Messages')

        if reset:
                st.write('Sure thing!')
                history = ChatHistory()
                st.session_state["history"] = history
                st.session_state["history"].add_system_message("You are a helpful assistant.") 
                print("completed reset")
                reset = False

        st.write("Upload RFP PDF file")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_file0")
    with col2:
        st.write("## RFP Chat")
        
        if uploaded_file is not None:
            # Display the PDF in an iframe
            pdf_bytes = uploaded_file.read()  # Read the PDF as bytes
            st.download_button("Download PDF", pdf_bytes, file_name="uploaded_pdf.pdf")

            # Convert to base64
            base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')

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
            result = await invoke_agent(prompt)
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(result)


asyncio.run(rfpsem())  