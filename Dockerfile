# Step 1: Selecting base image with Python + pip installed
FROM python:3.10.6-buster

# Step 2: Setting working directory within container
WORKDIR /app

# Step 3: Copying project files into container
COPY enviro_class /app/enviro_class
COPY api /app/api
COPY models /app/models
COPY requirements.txt /app/
COPY interface/streamlit_api.py /app/

# Step 4: Installing dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y python3-opencv

# STEP 5: Launching web server with Uvicorn
CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
