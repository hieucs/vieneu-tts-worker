FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Pin torch/torchvision to base image versions, then install vieneu with all deps
RUN TORCH_VER=$(python -c "import torch; print(torch.__version__)") && \
    TV_VER=$(python -c "import torchvision; print(torchvision.__version__)") && \
    pip install --no-cache-dir \
    "torch==${TORCH_VER}" \
    "torchvision==${TV_VER}" \
    torchaudio \
    "vieneu==1.2.3" \
    runpod>=1.7.0 && \
    echo "Installed with torch=${TORCH_VER}"

# Pre-download models
RUN python -c "\
from huggingface_hub import hf_hub_download, snapshot_download; \
snapshot_download('pnnbao-ump/VieNeu-TTS-0.3B'); \
hf_hub_download('neuphonic/distill-neucodec', 'pytorch_model.bin'); \
hf_hub_download('neuphonic/distill-neucodec', 'meta.yaml'); \
hf_hub_download('ntu-spml/distilhubert', 'config.json'); \
hf_hub_download('ntu-spml/distilhubert', 'model.safetensors'); \
hf_hub_download('ntu-spml/distilhubert', 'preprocessor_config.json'); \
print('Models downloaded')"

# Verify import works
RUN python -c "from vieneu import VieNeuTTS; print('VieNeuTTS import OK')"

COPY handler.py /app/handler.py

ENV CODEC_DEVICE=cuda
ENV BACKBONE_DEVICE=cuda
ENV BACKBONE_REPO=pnnbao-ump/VieNeu-TTS-0.3B

CMD ["python", "-u", "/app/handler.py"]
