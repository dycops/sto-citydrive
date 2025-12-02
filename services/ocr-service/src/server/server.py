import logging
import sys
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

import grpc
import time
import signal
from concurrent import futures

from paddleocr import PaddleOCR
import json
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "generated"))

from generated import ocr_pb2 as pb
from generated import ocr_pb2_grpc as pb_grpc


DEVICE = os.getenv("OCR_DEVICE", "gpu").lower()

Path("logs").mkdir(exist_ok=True)

file_handler = RotatingFileHandler(
    "logs/ocr_service.log",
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8"
)

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
file_handler.setFormatter(formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        file_handler
    ]
)

logger = logging.getLogger("ocr-grpc")


class OCRService(pb_grpc.OCRServiceServicer):
    def __init__(self):
        logger.info(f"Initializing PaddleOCR with device: {DEVICE}")

        try:
            self.ocr = PaddleOCR(
                lang="ru",
                device=DEVICE,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )

            # Warmup model (ускоряет первый запрос)
            dummy = np.zeros((100, 100, 3), dtype=np.uint8)
            self.ocr.predict(dummy)

            logger.info("PaddleOCR initialized successfully.")

        except Exception as e:
            logger.exception("Failed to initialize OCR:")
            raise e

    def Recognize(self, request, context):
        try:
            if not request.image:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Image field cannot be empty")
                return pb.OCRResponse(
                    success=False,
                    json="",
                    error="Empty image"
                )

            nparr = np.frombuffer(request.image, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Image decode error")
                return pb.OCRResponse(
                    success=False,
                    json="",
                    error="Unable to decode image"
                )

            results = self.ocr.predict(img)
            outputs = [r.json for r in results]

            return pb.OCRResponse(
                success=True,
                json=json.dumps(outputs, ensure_ascii=False),
                error=""
            )

        except Exception as e:
            logger.exception("OCR processing error:")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("OCR processing error")
            return pb.OCRResponse(success=False, json="", error=str(e))


_server = None


def serve():
    global _server
    
    logger.info("Starting gRPC server...")

    _server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=8),
        options=[
            ("grpc.keepalive_time_ms", 10000),
            ("grpc.keepalive_timeout_ms", 5000),
            ("grpc.max_send_message_length", 50 * 1024 * 1024),
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),
        ]
    )

    pb_grpc.add_OCRServiceServicer_to_server(OCRService(), _server)
    _server.add_insecure_port("[::]:50051")
    _server.start()

    logger.info("gRPC server started on port 50051")

    def shutdown_handler(signum, frame):
        logger.info(f"Received signal {signum}. Starting graceful shutdown...")
        _server.stop(grace=5)
        logger.info("Server stopped gracefully.")
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)

    try:
        while True:
            time.sleep(86400)
    except Exception as e:
        logger.exception(f"Server error: {e}")
        _server.stop(grace=5)
        sys.exit(1)
