# Use official Python image
FROM python:3.11-slim

# Install system dependencies for OpenCV and webcam
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all code and model
COPY . .

# Default command to run
CMD ["python", "person_detection.py"]
