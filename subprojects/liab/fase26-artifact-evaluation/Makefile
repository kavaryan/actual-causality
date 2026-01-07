.PHONY: all clean build-docker copy-sources download-image check-image

all: build-docker download-image

clean:
	rm -rf src/
	rm -f liab-artifact.tar
	docker rmi -f liab-artifact:latest 2>/dev/null || true

copy-sources:
	mkdir -p src/core src/subprojects/liab
	rsync -a --exclude='__pycache__/' ../../../core/ src/core/
	rsync -a --exclude='__pycache__/' --exclude='fase26-artifact-evaluation/' ../ src/subprojects/liab/

check-image:
	@if docker images -q liab-artifact:latest | grep -q .; then \
		echo "Image liab-artifact:latest already exists, skipping build"; \
		exit 0; \
	else \
		echo "Image liab-artifact:latest not found, will build"; \
		exit 1; \
	fi

build-docker: copy-sources
	@if $(MAKE) check-image 2>/dev/null; then \
		echo "Skipping build - image already exists"; \
	else \
		echo "Building Docker image..."; \
		docker build -t liab-artifact:latest .; \
	fi

download-image:
	@if [ ! -f liab-artifact.tar ]; then \
		echo "Saving Docker image to liab-artifact.tar..."; \
		docker save -o liab-artifact.tar liab-artifact:latest; \
	else \
		echo "Image archive liab-artifact.tar already exists"; \
	fi

smoke-test: build-docker
	docker run --rm liab-artifact:latest
