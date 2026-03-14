FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install llama-cpp-python pre-built wheel first
RUN pip install --no-cache-dir \
    llama-cpp-python==0.3.16 \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# Install vieneu without upgrading torch (base image already has correct torch+cuda)
RUN pip install --no-cache-dir --no-deps \
    "vieneu>=1.2.3" && \
    pip install --no-cache-dir \
    runpod>=1.7.0 \
    huggingface_hub \
    neucodec>=0.0.4 \
    librosa>=0.11.0 \
    phonemizer>=3.3.0 \
    perth>=0.2.0 \
    requests

# Pre-download models
RUN python -c "\
from huggingface_hub import hf_hub_download, snapshot_download; \
hf_hub_download('pnnbao-ump/VieNeu-TTS-0.3B-q4-gguf', 'VieNeu-TTS-0_3B-Q4_0.gguf'); \
hf_hub_download('neuphonic/distill-neucodec', 'pytorch_model.bin'); \
hf_hub_download('neuphonic/distill-neucodec', 'meta.yaml'); \
hf_hub_download('ntu-spml/distilhubert', 'config.json'); \
hf_hub_download('ntu-spml/distilhubert', 'model.safetensors'); \
hf_hub_download('ntu-spml/distilhubert', 'preprocessor_config.json'); \
snapshot_download('pnnbao-ump/VieNeu-TTS-0.3B'); \
print('All models downloaded')"

COPY handler.py /app/handler.py

ENV CODEC_DEVICE=cuda
ENV CUDA_MODULE_LOADING=LAZY

CMD ["python", "-u", "/app/handler.py"]
