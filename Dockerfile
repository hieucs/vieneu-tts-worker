FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install vieneu --no-deps, then its deps + transformers for GPU backbone
RUN pip install --no-cache-dir --no-deps "vieneu==1.2.3" && \
    pip install --no-cache-dir \
    runpod>=1.7.0 \
    huggingface_hub \
    neucodec>=0.0.4 \
    librosa>=0.11.0 \
    phonemizer>=3.3.0 \
    perth>=0.2.0 \
    transformers \
    accelerate \
    requests

# Pre-download transformers backbone (runs on GPU, no GGUF needed)
RUN python -c "\
from huggingface_hub import hf_hub_download, snapshot_download; \
snapshot_download('pnnbao-ump/VieNeu-TTS-0.3B'); \
hf_hub_download('neuphonic/distill-neucodec', 'pytorch_model.bin'); \
hf_hub_download('neuphonic/distill-neucodec', 'meta.yaml'); \
hf_hub_download('ntu-spml/distilhubert', 'config.json'); \
hf_hub_download('ntu-spml/distilhubert', 'model.safetensors'); \
hf_hub_download('ntu-spml/distilhubert', 'preprocessor_config.json'); \
print('Models downloaded')"

COPY handler.py /app/handler.py

ENV CODEC_DEVICE=cuda
ENV BACKBONE_DEVICE=cuda
ENV BACKBONE_REPO=pnnbao-ump/VieNeu-TTS-0.3B

CMD ["python", "-u", "/app/handler.py"]
