.PHONY: all clean build-docker copy-sources copy-requirements build-dockerfile

# Default target
all: build-docker

# Clean build artifacts
clean:
	rm -rf src/
	rm -f Dockerfile
	docker rmi -f liab-artifact:latest 2>/dev/null || true

# Copy source files
copy-sources:
	mkdir -p src/
	cp -r ../../../core src/
	cp -r ../ src/liab/

# Copy requirements file
copy-requirements:
	cp ../requirements.txt .

# Build Dockerfile
build-dockerfile: copy-requirements
	@echo "FROM python:3.9-slim" > Dockerfile
	@echo "" >> Dockerfile
	@echo "# Install system dependencies" >> Dockerfile
	@echo "RUN apt-get update && apt-get install -y \\" >> Dockerfile
	@echo "    build-essential \\" >> Dockerfile
	@echo "    && rm -rf /var/lib/apt/lists/*" >> Dockerfile
	@echo "" >> Dockerfile
	@echo "# Set working directory" >> Dockerfile
	@echo "WORKDIR /app" >> Dockerfile
	@echo "" >> Dockerfile
	@echo "# Copy requirements first for better caching" >> Dockerfile
	@echo "COPY requirements.txt ." >> Dockerfile
	@echo "RUN pip install --no-cache-dir -r requirements.txt" >> Dockerfile
	@echo "" >> Dockerfile
	@echo "# Copy source code" >> Dockerfile
	@echo "COPY src/ ." >> Dockerfile
	@echo "" >> Dockerfile
	@echo "# Set Python path" >> Dockerfile
	@echo "ENV PYTHONPATH=/app" >> Dockerfile
	@echo "" >> Dockerfile
	@echo "# Default command" >> Dockerfile
	@echo "CMD [\"python\", \"liab/smoke-test.py\"]" >> Dockerfile

# Build Docker image
build-docker: copy-sources build-dockerfile
	docker build -t liab-artifact:latest .

# Run the artifact
run:
	docker run --rm liab-artifact:latest
