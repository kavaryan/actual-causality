FROM python:3.9-slim

WORKDIR /artifact

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set Python path to include core modules
ENV PYTHONPATH=/artifact:/artifact/../../..

# Default command
CMD ["python", "smoke_test.py"]
