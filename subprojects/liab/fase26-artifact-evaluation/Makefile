.PHONY: all help clean build-docker copy-sources docker-ready smoke-test tests

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
	@if [ ! -d "src" ]; then \
		echo "copy-sources: creating src directory and copying sources"; \
		mkdir -p src/core src/subprojects/liab src/tests/test_subprojects/test_liab src/tests/test_core; \
		rsync -a --exclude='__pycache__/' --exclude='*.ipynb' ../../../core/ src/core/; \
		rsync -a --exclude='__pycache__/' --exclude='*.ipynb' --exclude='fase26-artifact-evaluation/' ../ src/subprojects/liab/; \
		rsync -a --exclude='__pycache__/' --exclude='*.ipynb' ../../../tests/test_core/ src/tests/test_core/; \
		rsync -a --exclude='__pycache__/' --exclude='*.ipynb' ../../../tests/test_subprojects/test_liab/ src/tests/test_subprojects/test_liab/; \
	else \
		echo "copy-sources: src directory already exists, skipping copy"; \
	fi

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

tests: docker-ready            # Run all tests in the container
	docker run --rm liab-artifact:latest python tests/test_core/test_bf.py
	docker run --rm liab-artifact:latest python tests/test_core/test_failure.py
	docker run --rm liab-artifact:latest python tests/test_core/test_scm.py
	docker run --rm liab-artifact:latest python tests/test_subprojects/test_liab/test_k_leg_liab.py

paper-results: docker-ready  # Run experiments for paper results
	docker run --rm -v ./images:/artifact/images liab-artifact:latest python subprojects/liab/case-study.py
