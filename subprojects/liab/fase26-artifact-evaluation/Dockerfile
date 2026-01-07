FROM python:3.9-slim

WORKDIR /artifact
COPY src/subprojects/liab/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ .

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH=./src

# Default command
CMD ["python", "src/subprojects/liab/smoke-test.py"]
