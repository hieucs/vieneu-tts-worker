FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    build-essential cmake git espeak-ng && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Step 1: Install torch+torchaudio+torchvision from PyTorch cu124 index
RUN pip install --no-cache-dir \
    torch torchaudio torchvision \
    --index-url https://download.pytorch.org/whl/cu124

# Step 2: Install llama-cpp-python pre-built wheel
RUN pip install --no-cache-dir \
    llama-cpp-python==0.3.16 \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# Step 3: Install vieneu (torch already installed, pip won't downgrade)
RUN pip install --no-cache-dir "vieneu==1.2.3" runpod>=1.7.0

# Verify everything works
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
