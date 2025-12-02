ARG DEVICE=gpu

FROM paddlepaddle/paddle:3.2.2-gpu-cuda11.8-cudnn8.9 AS base-gpu
FROM paddlepaddle/paddle:3.2.2 AS base-cpu
FROM base-${DEVICE} AS base

ENV TZ=Europe/Moscow \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

ARG DEVICE
ENV OCR_DEVICE=${DEVICE}

RUN apt update && apt install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    libglvnd0 \
    libgl1-mesa-glx \
    ffmpeg \
    fonts-dejavu-core \
    && apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m pip install --upgrade pip && \
    pip3 install --no-cache-dir \
        paddleocr==3.3.2 \
        opencv-python \
        pillow \
        grpcio==1.76.0 \
        grpcio-tools==1.76.0

COPY ./proto /app/proto
COPY ./src /app/src

RUN python3 -m grpc_tools.protoc \
    -I proto \
    --python_out=src/generated \
    --grpc_python_out=src/generated \
    proto/ocr.proto

EXPOSE 50051

CMD ["python3", "-m", "src.server"]
