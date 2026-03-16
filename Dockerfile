FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    build-essential cmake git espeak-ng && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    python -m pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# All pip via "python -m pip" to ensure Python 3.11

# Step 1: torch from cu124
RUN python -m pip install --no-cache-dir \
    torch torchaudio torchvision \
    --index-url https://download.pytorch.org/whl/cu124

# Step 2: llama-cpp-python
RUN python -m pip install --no-cache-dir \
    llama-cpp-python==0.3.16 \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# Step 3: vieneu --no-deps
RUN python -m pip install --no-cache-dir --no-deps "vieneu==1.2.3"

# Step 4: vieneu deps (no torch)
RUN python -m pip install --no-cache-dir \
    phonemizer>=3.3.0 \
    neucodec>=0.0.4 \
    librosa>=0.11.0 \
    perth>=0.2.0 \
    transformers \
    accelerate \
    torchtune \
    local_attention \
    datasets \
    onnxruntime \
    requests \
    runpod>=1.7.0 \
    huggingface_hub

# Verify
RUN python -c "import torch; print(f'torch={torch.__version__}')" && \
    python -c "from vieneu import VieNeuTTS; print('VieNeuTTS OK')"

# Pre-download models
RUN python -c "\
from huggingface_hub import hf_hub_download, snapshot_download; \
snapshot_download('pnnbao-ump/VieNeu-TTS-0.3B'); \
hf_hub_download('neuphonic/distill-neucodec', 'pytorch_model.bin'); \
hf_hub_download('neuphonic/distill-neucodec', 'meta.yaml'); \
hf_hub_download('ntu-spml/distilhubert', 'config.json'); \
hf_hub_download('ntu-spml/distilhubert', 'model.safetensors'); \
hf_hub_download('ntu-spml/distilhubert', 'preprocessor_config.json'); \
print('Models OK')"

COPY handler.py /app/handler.py

ENV CODEC_DEVICE=cuda
ENV BACKBONE_DEVICE=cuda
ENV BACKBONE_REPO=pnnbao-ump/VieNeu-TTS-0.3B

CMD ["python", "-u", "/app/handler.py"]
