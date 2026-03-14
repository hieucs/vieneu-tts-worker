FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install llama-cpp-python from pre-built wheel (much faster than building from source)
RUN pip install --no-cache-dir \
    llama-cpp-python==0.3.16 \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# Install remaining deps
RUN pip install --no-cache-dir \
    runpod>=1.7.0 \
    "vieneu>=1.2.3" \
    numpy==1.26.4 \
    huggingface_hub

# Pre-download models (faster cold start, no download at runtime)
RUN python -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download('pnnbao-ump/VieNeu-TTS-0.3B-q4-gguf', 'VieNeu-TTS-0_3B-Q4_0.gguf'); \
hf_hub_download('neuphonic/distill-neucodec', 'pytorch_model.bin'); \
hf_hub_download('neuphonic/distill-neucodec', 'meta.yaml'); \
hf_hub_download('ntu-spml/distilhubert', 'config.json'); \
hf_hub_download('ntu-spml/distilhubert', 'model.safetensors'); \
hf_hub_download('ntu-spml/distilhubert', 'preprocessor_config.json'); \
print('All models downloaded')"

# Also pre-download the non-GGUF transformers backbone as fallback
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('pnnbao-ump/VieNeu-TTS-0.3B'); \
print('Transformers backbone downloaded')"

COPY handler.py /app/handler.py

ENV CODEC_DEVICE=cuda
ENV CUDA_MODULE_LOADING=LAZY

CMD ["python", "-u", "/app/handler.py"]
