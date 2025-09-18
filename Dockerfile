#1. Base image with python
FROM python:3.11-slim

#2. Set working directory
WORKDIR /app

#3.Copy dependencies files and install
RUN pip install --no-cache-dir -r requirements.txt

#4.Copy code and YOLO model into container
COPY . .

#5. Default command to run app
CMD ["python", "person_detection.py"]