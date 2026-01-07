.PHONY: all clean build-docker copy-sources copy-requirements build-dockerfile

all: build-docker

clean:
	rm -rf src/
	docker rmi -f liab-artifact:latest 2>/dev/null || true

copy-sources:
	mkdir -p src/core src/subprojects/liab
	rsync -a --exclude='__pycache__/' ../../../core/ src/core/
	rsync -a --exclude='__pycache__/' --exclude='fase26-artifact-evaluation/' ../ src/subprojects/liab/
	
build-docker: copy-sources
	docker build -t liab-artifact:latest .

smoke-test:
	docker run --rm liab-artifact:latest
