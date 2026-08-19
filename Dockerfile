# MRIxFields2026 Task 2 — Docker Submission
#
# Build:
#   docker build -t mrixfields-task2 .
#
# Test locally (requires NVIDIA Docker runtime):
#   docker run --gpus all \
#       -v /path/to/input:/input:ro \
#       -v /path/to/output:/output \
#       mrixfields-task2
#
# Before building:
#   1. Model weights must be at Baseline/weights/*.pth (committed in Git)
#   2. (Optional) Place bbox.json at Baseline/scripts/bbox.json for faster
#      inference. Without it, inference processes full 364x436 slices instead
#      of the cropped brain region (~3x slower but produces identical results).

FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

RUN pip install --no-cache-dir \
    nibabel>=5.0.0 \
    tqdm>=4.65.0

WORKDIR /app

# NAFNet model architecture
COPY Baseline/mrixfields/models/nafnet.py /app/nafnet.py

# Trained weights: 12 checkpoints, one per (target_field x modality)
COPY Baseline/weights/*.pth /app/weights/

# bbox.json: copy from scripts/ if available, skip otherwise
COPY Baseline/scripts/ /tmp/build_scripts/
RUN if [ -f /tmp/build_scripts/bbox.json ]; then \
        cp /tmp/build_scripts/bbox.json /app/bbox.json && \
        echo "bbox.json copied to /app/bbox.json"; \
    else \
        echo "bbox.json not found — inference will use full-volume fallback"; \
    fi && \
    rm -rf /tmp/build_scripts/

# Inference entrypoint
COPY inference.py /app/inference.py

ENTRYPOINT ["python", "/app/inference.py"]
