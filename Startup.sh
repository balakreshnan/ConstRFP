#!/bin/bash
# Set default port if not provided
PORT=${PORT:-8501}
# Start the Streamlit app
streamlit run app.py --server.port $PORT