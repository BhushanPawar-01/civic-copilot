#!/bin/bash
set -e

echo "Starting FastAPI backend..."
cd backend
uvicorn main:app --host 127.0.0.1 --port 8000 &

echo "Waiting for backend to be ready..."
sleep 10

echo "Starting Streamlit frontend..."
cd ../frontend
streamlit run streamlit_app.py \
  --server.port=8501 \
  --server.address=0.0.0.0
