# Use Python 3.10 base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy required files
COPY requirements.txt .
COPY api/ ./api/
COPY models/ ./models/
# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Expose FastAPI default port
EXPOSE 8000

# Run the API app
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

