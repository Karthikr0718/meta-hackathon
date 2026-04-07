FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV API_BASE_URL="https://router.huggingface.co/v1"
ENV MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
ENV HF_TOKEN=""

# Run inference
CMD ["python", "app.py"]
