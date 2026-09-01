FROM nvidia/cuda:13.0.2-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       ca-certificates \
       cmake \
       git \
       ninja-build \
       python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . .

RUN cmake --preset ci \
    && cmake --build --preset ci-build

CMD ["bash"]
