# Step 1: Selecting base image with Python + pip installed
FROM python:3.10.6-buster

# Step 2: Copying project files into container
COPY enviro_class /app/enviro_class
COPY api /app/api
COPY requirements.txt /app/

# Step 3: Installing dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Step 4: Launching web server with Uvicorn
CMD uvicorn enviroclass.api.fast:app --host 0.0.0.0 --port $PORT
