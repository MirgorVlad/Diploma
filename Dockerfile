FROM python:3.8-slim

# Install necessary packages
RUN pip install seldon-core joblib scikit-learn flask mlflow pandas

# Create the model directory
RUN mkdir -p /mnt/models

# Copy the application code
COPY app.py /app/app.py

# Set the working directory
WORKDIR /app

# Expose the port for Flask app
EXPOSE 5001

# Command to run Flask app
CMD ["python", "app.py"]
