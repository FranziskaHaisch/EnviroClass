# Step 1: Selecting base image with Python + pip installed
FROM python:3.10.6-buster

# Step 2: Setting working directory within container
WORKDIR /app

# Step 3: Copying project files into container
COPY enviro_class /app/enviro_class
COPY api /app/api
COPY models /app/models
COPY requirements.txt /app/

# Step 4: Installing dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Step 5: Exposing both APIs (wildfire + environment)
EXPOSE 8001 8002

# STEP 6: Launching web server with Uvicorn (starting both APIs)
CMD uvicorn api.wildfire:app --host 0.0.0.0 --port 8001 & \
    uvicorn api.environment:app --host 0.0.0.0 --port 8002
