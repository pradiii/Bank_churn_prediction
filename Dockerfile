# Set up a Python 3.10 slim image.
FROM python:3.10-slim-buster

# Update the package lists and install necessary development tools.
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip.
RUN pip install --upgrade pip

# Copy the local project files to the container.
COPY . /app

# Install Python dependencies from requirements.txt.
RUN pip install --no-cache-dir --upgrade -r app/requirements.txt

#Expose the port streamlit run on
EXPOSE 8501

# Set the working directory to /app.
WORKDIR /app
 
# Add /app to the PYTHONPATH.
ENV PYTHONPATH "${PYTHONPATH}:/app"

# Install the package in editable mode when the container starts.
CMD pip install -e .

#Run streamlit when the container launches.
CMD ["streamlit","run", "app_stream_lit.py"]
