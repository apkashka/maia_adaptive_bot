FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY templates/ templates/
COPY static/ static/

# Copy maia2 module (inner package with local modifications)
COPY maia2/maia2/ maia2/

# Create maia2_models dir (model will be downloaded on first run)
RUN mkdir -p maia2_models

# Expose port for HF Spaces
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
