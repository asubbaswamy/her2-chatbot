# Pre-pull the ollama models.
# that way users can run chatbot out of the box
FROM ollama/ollama:latest AS model-puller

# Start Ollama just to pull models
# for a slimmer demo, just using 3.2:1b
RUN ollama serve & \
    sleep 10 && \
    # ollama pull llama3.2 && \
    ollama pull llama3.2:1b && \
    pkill ollama && \
    sleep 5

# build out our container
FROM python:3.11.11

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# install ollama on the container
RUN curl -fsSL https://ollama.com/install.sh | sh

# get the models we already pulled
COPY --from=model-puller /root/.ollama /root/.ollama

# Run Ollama in the background
# Pull the models we use
# RUN echo '#!/bin/bash\n\
# # Start Ollama service\n\
# ollama serve > /var/log/ollama.log 2>&1 &\n\
# OLLAMA_PID=$!\n\
# \n\
# # Wait for Ollama to start\n\
# echo "Waiting for Ollama to start..."\n\
# sleep 8\n\
# \n\
# # Pull models\n\
# echo "Pulling models..."\n\
# ollama pull llama3.2\n\
# ollama pull llama3.2:1b\n\'

RUN echo '#!/bin/bash\n\
# Start Ollama service\n\
ollama serve > /var/log/ollama.log 2>&1 &\n\
OLLAMA_PID=$!\n\
\n\
# Wait for Ollama to start\n\
echo "Waiting for Ollama to start..."\n\
sleep 5\n\
\n\
# Run the command\n\
echo "Running: $@"\n\
exec "$@"\n' > /run.sh && chmod +x /run.sh

# Copy code and data
COPY . .

# Set environment variables
# NONE

# run the chatbot
ENTRYPOINT ["/run.sh"]
CMD ["python", "src/chatbot.py", "--model", "llama3.2:1b"]

# could create a script for running it...
# e.g., run.sh
# ENTRYPOINT ["/run.sh"]