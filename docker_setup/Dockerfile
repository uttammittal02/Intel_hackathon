# Use an official Python runtime as a base image
FROM intel/deep-learning:latest-py3.10

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies for the app (Gradio, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that the Gradio app will run on (default is 7860)
EXPOSE 7860

ENV GRADIO_SERVER_NAME="0.0.0.0"

# Command to run the Gradio app
CMD ["python", "app.py"]
