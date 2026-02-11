FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY requirements.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY main.py ./
COPY .env.example .env

# Create directories for data persistence
RUN mkdir -p /app/snapshots

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose port for potential API (future extension)
EXPOSE 8000

# Default command: run interactive CLI
CMD ["python", "main.py"]
