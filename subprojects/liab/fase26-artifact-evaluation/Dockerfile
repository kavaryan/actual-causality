FROM python:3.9-slim

WORKDIR /artifact
COPY src/liab/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ .

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH=/artifact

# Default command
CMD ["python", "liab/smoke-test.py"]
