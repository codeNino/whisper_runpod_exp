# -------- Config (override via command line) --------
IMAGE_NAME ?= codenino/whisper_runpod
TAG ?= latest
DOCKERFILE ?= Dockerfile
PLATFORM ?= linux/amd64

# -------- Targets --------
.PHONY: build push build-push

build:
	docker build \
		-f $(DOCKERFILE) \
		-t $(IMAGE_NAME):$(TAG) \
		--platform $(PLATFORM) \
		.

push:
	docker push $(IMAGE_NAME):$(TAG)

build-push: build push