FROM python:3.13.5-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list
COPY requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Create .env file with defaults (HuggingFace Spaces compatible)
RUN echo 'ENVIRONMENT=production\n\
LOG_LEVEL=INFO\n\
BACKEND_PORT=8000\n\
FRONTEND_PORT=8501' > .env

# Make startup script executable
RUN chmod +x start.sh

# Expose Streamlit port (HF requirement)
EXPOSE 8501

# Healthcheck for Streamlit
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Start both FastAPI and Streamlit
ENTRYPOINT ["./start.sh"]
