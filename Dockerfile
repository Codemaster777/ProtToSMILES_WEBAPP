# Use Python 3.8 as the base image
FROM python:3.8

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port (Hugging Face uses 7860 by default)
EXPOSE 7860

# Run the app with Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
