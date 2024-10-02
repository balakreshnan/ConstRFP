# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install build tools, libraries, and ffmpeg
RUN apt-get update && apt-get install -y \
    gcc \
    libasound2-dev \
    portaudio19-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

COPY SAMPLE_PLAN-PROF.pdf /app/SAMPLE_PLAN-PROF.pdf
#RUN find . -name "*.pdf" -delete

# Make the startup script executable
RUN chmod +x Startup.sh

# Expose the port Streamlit will run on
EXPOSE 8501

# Run the startup script
CMD ["./Startup.sh"]