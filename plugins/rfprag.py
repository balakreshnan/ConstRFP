import PyPDF2
from openai import AzureOpenAI
from semantic_kernel.functions.kernel_function_decorator import kernel_function
import requests
from bs4 import BeautifulSoup
from collections import defaultdict
import io
import os
from typing import Annotated

class rfpchat:    
    @kernel_function(
        name="summarise_rfp",
        description="Summarize rfp infomration.",
    )
    def summarise_rfp(
        self,
        query: str,
        pdf_bytes: bytes = None,
    ) -> Annotated[str, "the output is a string"]:
        """Content from web page to process further."""
        summary = ""
        
        #content = extracttextfrompdf(pdf_bytes)
        print("RFP summarizer chat")
        if query:
            summary = extractrfpinformation(query, pdf_bytes)
            return summary
        else:
            return None
        return summary
    
def extracttextfrompdf(pdf_bytes):
    # Extract text from PDF
    rfttext = ""

    if pdf_bytes:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        num_pages = len(reader.pages)
        # Extract and display text from the first page
        if num_pages > 0:
            for page_num in range(num_pages):
                page = reader.pages[page_num]  # Get each page
                text = page.extract_text()  # Extract text from the page
                rfttext += f"### Page {page_num + 1}\n{text}\n\n"  # Accumulate text from each page
    return rfttext

def extractrfpinformation(query, pdf_bytes):
    returntxt = ""

    rfttext = ""
    client = AzureOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2024-05-01-preview"
    )

    model_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

    if pdf_bytes:
        rfttext = extracttextfrompdf(pdf_bytes)

    # print('RFP Text:', rfttext)

    message_text = [
    {"role":"system", "content":f"""You are RFP AI agent. Be politely, and provide positive tone answers.
     Based on the question do a detail analysis on RFP information and provide the best answers.
     Here is the RFT text tha was provided:
     {rfttext}
     please provide information based on rfp provided.
     Only provide answers from the content of the RFP.
     If not sure, ask the user to provide more information."""}, 
    {"role": "user", "content": f"""{query}. Respond Only the answers with details on why the decision was made."""}]

    response = client.chat.completions.create(
        model= model_name, #"gpt-4-turbo", # model = "deployment_name".
        messages=message_text,
        temperature=0.0,
        top_p=0.0,
        seed=105,
    )

    returntxt = response.choices[0].message.content
    return returntxt