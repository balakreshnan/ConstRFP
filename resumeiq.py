import os
from openai import AzureOpenAI
import gradio as gr
from dotenv import dotenv_values
import time
from datetime import timedelta
import json
import streamlit as st
from PIL import Image
import base64
import requests
import io
import autogen
from typing import Optional
from typing_extensions import Annotated
from streamlit import session_state as state
import azure.cognitiveservices.speech as speechsdk
from audiorecorder import audiorecorder
import pyaudio
import wave
import PyPDF2
import docx
import fitz  # PyMuPDF
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import numpy as np
from streamlit_quill import st_quill

config = dotenv_values("env.env")

css = """
.container {
    height: 75vh;
}
"""

client = AzureOpenAI(
  azure_endpoint = config["AZURE_OPENAI_ENDPOINT_VISION"], 
  api_key=config["AZURE_OPENAI_KEY_VISION"],  
  api_version="2024-05-01-preview"
  #api_version="2024-02-01"
  #api_version="2023-12-01-preview"
  #api_version="2023-09-01-preview"
)

#model_name = "gpt-4-turbo"
#model_name = "gpt-35-turbo-16k"
model_name = "gpt-4o-g"

search_endpoint = config["AZURE_AI_SEARCH_ENDPOINT"]
search_key = config["AZURE_AI_SEARCH_KEY"]
search_index=config["AZURE_AI_SEARCH_INDEX1"]
SPEECH_KEY = config['SPEECH_KEY']
SPEECH_REGION = config['SPEECH_REGION']
SPEECH_ENDPOINT = config['SPEECH_ENDPOINT']

citationtxt = ""


def extracttextfrompdf(pdf_bytes):
    returntxt = ""

    if pdf_bytes:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        num_pages = len(reader.pages)
        st.write(f"Number of pages in the PDF: {num_pages}")
        # Extract and display text from the first page
        if num_pages > 0:
            page = reader.pages[0]  # Get the first page
            text = page.extract_text()  # Extract text from the page
            returntxt = text

    return returntxt

def processpdfwithprompt(user_input1, selected_optionmodel1, selected_optionsearch):
    returntxt = ""
    citationtxt = ""
    message_text = [
    {"role":"system", "content":"""you are provided with instruction on what to do. Be politely, and provide positive tone answers. 
     answer only from data source provided. unable to find answer, please respond politely and ask for more information.
     Extract Title content from the document. Show the Title as citations which is provided as Title: as [doc1] [doc2].
     Please add citation after each sentence when possible in a form "(Title: citation)".
     Be polite and provide posite responses. If user is asking you to do things that are not specific to this context please ignore."""}, 
    {"role": "user", "content": f"""{user_input1}"""}]

    response = client.chat.completions.create(
        model= selected_optionmodel1, #"gpt-4-turbo", # model = "deployment_name".
        messages=message_text,
        temperature=0.0,
        top_p=1,
        seed=105,
        extra_body={
        "data_sources": [
            {
                "type": "azure_search",
                "parameters": {
                    "endpoint": search_endpoint,
                    "index_name": search_index,
                    "authentication": {
                        "type": "api_key",
                        "key": search_key
                    },
                    "include_contexts": ["citations"],
                    "top_n_documents": 5,
                    "query_type": selected_optionsearch,
                    "semantic_configuration": "my-semantic-config",
                    "embedding_dependency": {
                        "type": "deployment_name",
                        "deployment_name": "text-embedding-ada-002"
                    },
                    "fields_mapping": {
                        "content_fields": ["chunk"],
                        "vector_fields": ["text_vector"],
                        "title_field": "title",
                        "url_field": "title",
                        "filepath_field": "title",
                        "content_fields_separator": "\n",
                    }
                }
            }
        ]
    }
    )
    #print(response.choices[0].message.context)

    returntxt = response.choices[0].message.content + "\n<br>"

    json_string = json.dumps(response.choices[0].message.context)

    parsed_json = json.loads(json_string)

    # print(parsed_json)

    if parsed_json['citations'] is not None:
        returntxt = returntxt + f"""<br> Citations: """
        for row in parsed_json['citations']:
            #returntxt = returntxt + f"""<br> Title: {row['filepath']} as {row['url']}"""
            #returntxt = returntxt + f"""<br> [{row['url']}_{row['chunk_id']}]"""
            returntxt = returntxt + f"""<br> <a href='{row['url']}' target='_blank'>[{row['url']}_{row['chunk_id']}]</a>"""
            citationtxt = citationtxt + f"""<br><br> Title: {row['title']} <br> URL: {row['url']} 
            <br> Chunk ID: {row['chunk_id']} 
            <br> Content: {row['content']} 
            <br> ------------------------------------------------------------------------------------------ <br>\n"""

    return returntxt, citationtxt

def extractresumeresults(user_input1, selected_optionmodel1, pdf_bytes, selected_optionsearch):
    returntxt = ""

    rfttext = ""

    if pdf_bytes:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        num_pages = len(reader.pages)
        st.write(f"Number of pages in the PDF: {num_pages}")
        # Extract and display text from the first page
        if num_pages > 0:
            for page_num in range(num_pages):
                page = reader.pages[page_num]  # Get each page
                text = page.extract_text()  # Extract text from the page
                rfttext += f"### Page {page_num + 1}\n{text}\n\n"  # Accumulate text from each page

    # print('RFP Text:', rfttext)

    #dstext = processpdfwithprompt(user_input1, selected_optionmodel1, selected_optionsearch)

    message_text = [
    {"role":"system", "content":f"""You are Resume AI agent. Be politely, and provide positive tone answers.
     Analze the resume for the resource requested. Provide insights on what is missing in the resume.
     Provide recommendations on what can be added to the resume.
     Be creative and provide insights on what can be added to the resume.
     Provide Strenghts, Area of improvements, Recommendations, and Insights.
     Here is the Resume content provided:
     {rfttext}

     if the question is outside the bounds of the Resume, Let the user know answer might be relevant for Resume provided.
     If not sure, ask the user to provide more information."""}, 
    {"role": "user", "content": f"""{user_input1}. Provide resume insights and format well."""}]

    response = client.chat.completions.create(
        model= selected_optionmodel1, #"gpt-4-turbo", # model = "deployment_name".
        messages=message_text,
        temperature=0.0,
        top_p=0.0,
        seed=105,
    )

    returntxt = response.choices[0].message.content
    return returntxt

def createresumeresults(user_input1, selected_optionmodel1, pdf_bytes, selected_optionsearch):
    returntxt = ""

    rfttext = ""

    if pdf_bytes:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        num_pages = len(reader.pages)
        st.write(f"Number of pages in the PDF: {num_pages}")
        # Extract and display text from the first page
        if num_pages > 0:
            for page_num in range(num_pages):
                page = reader.pages[page_num]  # Get each page
                text = page.extract_text()  # Extract text from the page
                rfttext += f"### Page {page_num + 1}\n{text}\n\n"  # Accumulate text from each page

    # print('RFP Text:', rfttext)

    #dstext = processpdfwithprompt(user_input1, selected_optionmodel1, selected_optionsearch)

    message_text = [
    {"role":"system", "content":f"""You are Resume AI agent. Be politely, and provide positive tone answers.
     Analze the resume for the resource requested. Create Strenghts, Area of improvements, Recommendations
     For Each recommendation create content that can be added to the resume.
     Be creative and provide insights on what can be added to the resume.
     Create content for recomemndations that can be used in resume.
     Here is the Resume content provided:
     {rfttext}

     if the question is outside the bounds of the Resume, Let the user know answer might be relevant for Resume provided.
     If not sure, ask the user to provide more information."""}, 
    {"role": "user", "content": f"""{user_input1}. Create recommendation and Area of improvements content."""}]

    response = client.chat.completions.create(
        model= selected_optionmodel1, #"gpt-4-turbo", # model = "deployment_name".
        messages=message_text,
        temperature=0.0,
        top_p=0.0,
        seed=105,
    )

    returntxt = response.choices[0].message.content
    return returntxt

def createfinalresumeresults(user_input1, selected_optionmodel1, pdf_bytes, selected_optionsearch):
    returntxt = ""

    rfttext = ""

    if pdf_bytes:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        num_pages = len(reader.pages)
        st.write(f"Number of pages in the PDF: {num_pages}")
        # Extract and display text from the first page
        if num_pages > 0:
            for page_num in range(num_pages):
                page = reader.pages[page_num]  # Get each page
                text = page.extract_text()  # Extract text from the page
                rfttext += f"### Page {page_num + 1}\n{text}\n\n"  # Accumulate text from each page

    # print('RFP Text:', rfttext)

    #dstext = processpdfwithprompt(user_input1, selected_optionmodel1, selected_optionsearch)

    message_text = [
    {"role":"system", "content":f"""You are Resume AI agent. Be politely, and provide positive tone answers.
     Analze the resume for the resource requested. Create Strenghts, Area of improvements, Recommendations
     For Each recommendation create content that can be added to the resume.
     Be creative and provide insights on what can be added to the resume.
     Create content for recomemndations that can be used in resume.
     Here is the Resume content provided:
     {rfttext}

     if the question is outside the bounds of the Resume, Let the user know answer might be relevant for Resume provided.
     If not sure, ask the user to provide more information."""}, 
    {"role": "user", "content": f"""{user_input1}. Create the complete resume with recommendation and Area of improvements."""}]

    response = client.chat.completions.create(
        model= selected_optionmodel1, #"gpt-4-turbo", # model = "deployment_name".
        messages=message_text,
        temperature=0.0,
        top_p=0.0,
        seed=105,
    )

    returntxt = response.choices[0].message.content
    return returntxt

def speech_to_text_extract(text, selected_optionmodel1):
    returntxt = ""

    start_time = time.time()

    message_text = [
    {"role":"system", "content":"""You are a Lanugage AI Agent, based on the text provided, extract intent and also the value provided.
     For example change sugar from 5g to 10g. change sugar to 10g.
     Provide the extracted ingredient and value to update only.     
     """}, 
    {"role": "user", "content": f"""Content: {text}."""}]

    response = client.chat.completions.create(
        model= selected_optionmodel1, #"gpt-4-turbo", # model = "deployment_name".
        messages=message_text,
        temperature=0.0,
        top_p=1,
        seed=105,
   )

    returntxt = response.choices[0].message.content

def recognize_from_microphone():
    returntxt = ""
    # This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
    speech_config = speechsdk.SpeechConfig(subscription=config['SPEECH_KEY'], region=config['SPEECH_REGION'])
    speech_config.speech_recognition_language="en-US"

    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    print("Speak into your microphone.")
    speech_recognition_result = speech_recognizer.recognize_once_async().get()

    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(speech_recognition_result.text))
        st.write(f"Recognized: {speech_recognition_result.text}")
        speech_to_text_extract(speech_recognition_result.text, "gpt-4o-g")
        returntxt = speech_recognition_result.text
    elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(speech_recognition_result.no_match_details))
    elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_recognition_result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
            print("Did you set the speech resource key and region values?")
    return returntxt

if 'resumecontent' not in st.session_state:
    st.session_state.resumecontent = 'I want to look for Chief Technolog officer profiles'

def update_quill_resumecontent(new_content):  
    # Append new content to the existing content changed
    st.session_state.resumecontent += new_content

def resumeiq():
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

    tabs = st.tabs(["Upload Resume", "Resume Insights", "Resume Recommendation", "Final Resume", "Create Word"
                    ])
    
    with tabs[0]:
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_file0")
        if uploaded_file is not None:
            # Display the PDF in an iframe
            pdf_bytes = uploaded_file.read()  # Read the PDF as bytes
            st.download_button("Download PDF", pdf_bytes, file_name="uploaded_pdf.pdf")

            # Convert to base64
            base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
            # Embedding PDF using an HTML iframe
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="1000" height="700" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
            # Save the PDF file to the current folder
            file_path = os.path.join(os.getcwd(), uploaded_file.name)  # Save in the current directory
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())  # Write the uploaded file to disk
            
            # Display the path where the file is stored
            # st.write(f"File saved to: {file_path}")
            temp_file_path = file_path
    with tabs[1]:
        if pdf_bytes:
            user_input1 = st.text_area("Enter your question", "I want to look for Chief Technolog officer profiles?")
            #selected_optionsearch = st.selectbox("Select an Search Type:", ["semantic", "full"])
            if st.button("Get Resume Summary"):
                resulttxt = extractresumeresults(user_input1, selected_optionmodel1, pdf_bytes, "semantic")
                st.markdown(resulttxt, unsafe_allow_html=True)
    with tabs[2]:
        if pdf_bytes:
            #user_input1 = st.text_area("Enter your question", "I want to look for Chief Technolog officer profiles?")
            #selected_optionsearch = st.selectbox("Select an Search Type:", ["semantic", "full"])
            if st.button("Get Resume Recommendation"):
                resulttxt = createresumeresults(user_input1, selected_optionmodel1, pdf_bytes, "semantic")
                st.markdown(resulttxt, unsafe_allow_html=True)
    with tabs[3]:
        if pdf_bytes:
            #user_input1 = st.text_area("Enter your question", "I want to look for Chief Technolog officer profiles?")
            #selected_optionsearch = st.selectbox("Select an Search Type:", ["semantic", "full"])
            
            if st.button("Get Final Resume"):
                resulttxt = createfinalresumeresults(user_input1, selected_optionmodel1, pdf_bytes, "semantic")
                # st.markdown(resulttxt, unsafe_allow_html=True)
                # st_quill(resulttxt, placeholder="Enter your rich text...",    key="editor1")
                if "quill_rfpcontent" not in st.session_state:
                    st.session_state.quill_rfpcontent = ""                
                update_quill_resumecontent(resulttxt)
            resumecontent = st_quill(st.session_state.resumecontent, placeholder="Enter your rich text...",    key="editorr1")