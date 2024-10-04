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

