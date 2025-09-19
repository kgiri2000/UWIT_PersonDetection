# Use official Python slim image
FROM python:3.11-slim

# Install system dependencies needed for OpenCV + sound
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    alsa-utils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all code
COPY . .

# Run the app
CMD ["python", "person_detection.py"]
