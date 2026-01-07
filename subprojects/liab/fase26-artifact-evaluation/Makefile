.PHONY: all help clean build-docker copy-sources docker-ready smoke-test

all: help                     # Show help

help:                          # Show this help message
	@awk 'BEGIN {FS = ":.*#"} \
	/^[a-zA-Z0-9_-]+:.*#/ { \
		printf "  %-16s %s\n", $$1, $$2 \
	}' $(MAKEFILE_LIST)

clean:                         # Remove build artifacts and docker image
	rm -rf src/
	rm -f liab-artifact.tar
	docker rmi -f liab-artifact:latest 2>/dev/null || true

copy-sources:                  # Copy sources from full repo into docker build context (DON'T USE DIRECTLY)
	mkdir -p src/core src/subprojects/liab src/tests/test_subprojects/test_liab src/tests/test_core
	rsync -a --exclude='__pycache__/' ../../../core/ src/core/
	rsync -a --exclude='__pycache__/' --exclude='fase26-artifact-evaluation/' ../ src/subprojects/liab/
	rsync -a --exclude='__pycache__/' ../../../tests/test_core/ src/tests/test_core/
	rsync -a --exclude='__pycache__/' ../../../tests/test_subprojects/test_liab/ src/tests/test_subprojects/test_liab/
	cp ./requirements.txt ./src/subprojects/liab/requirements.txt

build-docker: copy-sources     # Build docker image and export tarball
	docker build -t liab-artifact:latest .
	docker save -o liab-artifact.tar liab-artifact:latest

docker-ready:                  # Ensure image exists in docker daemon
	@set -e; \
	if docker image inspect liab-artifact:latest >/dev/null 2>&1; then \
		echo "docker-ready: liab-artifact:latest already present"; \
	elif [ -f "liab-artifact.tar" ]; then \
		echo "docker-ready: loading liab-artifact.tar"; \
		docker load -i liab-artifact.tar >/dev/null; \
	else \
		echo "docker-ready: building liab-artifact:latest"; \
		$(MAKE) build-docker; \
	fi

smoke-test: docker-ready        # Run container as a sanity check
	docker run --rm liab-artifact:latest
