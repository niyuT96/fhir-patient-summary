FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/ ./src/
COPY data/ ./data/

# Expose Gradio port
EXPOSE 7860

CMD ["python", "-m", "src.start"]
