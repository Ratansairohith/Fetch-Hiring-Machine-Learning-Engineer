FROM python:3.7

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install necessary libraries and packages
RUN pip install --no-cache-dir xgboost gradio pandas plotly kaleido pillow

# Run the Gradio app
CMD ["python", "inference.py"]
