FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install torchaudio matching base torch (2.4.x + cu124)
RUN pip install --no-cache-dir torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install vieneu without deps (avoid torch version conflicts)
RUN pip install --no-cache-dir --no-deps "vieneu==1.2.3"

# Install all vieneu runtime deps manually
RUN pip install --no-cache-dir \
    runpod>=1.7.0 \
    phonemizer>=3.3.0 \
    neucodec>=0.0.4 \
    librosa>=0.11.0 \
    perth>=0.2.0 \
    "transformers>=4.46" \
    accelerate \
    local_attention \
    torchtune \
    requests \
    huggingface_hub

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

# Fix torchvision if broken by transformers upgrade
RUN pip install --no-cache-dir torchvision --index-url https://download.pytorch.org/whl/cu124 || true

# Verify vieneu imports
RUN python -c "from vieneu import VieNeuTTS; print('VieNeuTTS OK')"

COPY handler.py /app/handler.py

ENV CODEC_DEVICE=cuda
ENV BACKBONE_DEVICE=cuda
ENV BACKBONE_REPO=pnnbao-ump/VieNeu-TTS-0.3B

CMD ["python", "-u", "/app/handler.py"]
