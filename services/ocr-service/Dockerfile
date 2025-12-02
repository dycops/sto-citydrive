FROM paddlepaddle/paddle:3.2.2-gpu-cuda11.8-cudnn8.9

ENV TZ=Europe/Moscow \
    DEBIAN_FRONTEND=noninteractive

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
    pip3 install \
        paddleocr==3.3.2 \
        opencv-python \
        pillow \
        grpcio==1.76.0 \
        grpcio-tools==1.76.0

COPY ./server /app/server
COPY ./.paddlex/official_models /root/.paddlex

RUN python3 -m grpc_tools.protoc \
    -I server/proto \
    --python_out=server \
    --grpc_python_out=server \
    server/proto/ocr.proto

EXPOSE 50051

RUN chmod +x server/start.sh

CMD ["bash", "server/start.sh"]
