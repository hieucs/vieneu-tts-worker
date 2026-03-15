"""RunPod Serverless Handler for VieNeu-TTS.

Deploy this on RunPod as a serverless endpoint.
It handles TTS inference requests and returns base64-encoded WAV audio.

Environment variables (set in RunPod template):
  CODEC_DEVICE: "cuda" (default) or "cpu"
"""
import base64
import io
import os
import struct
import sys
import logging
import tempfile

import numpy as np

logging.basicConfig(level=logging.INFO, format="[TTS] %(message)s")
logger = logging.getLogger(__name__)

# Global model instance
tts_model = None


def _numpy_to_wav(audio_np, sample_rate=24000):
    """Convert numpy float32 array to WAV bytes."""
    audio_np = np.clip(audio_np, -1.0, 1.0)
    pcm = (audio_np * 32767).astype(np.int16).tobytes()
    ch, sw = 1, 2
    data_size = len(pcm)
    wav = bytearray()
    wav.extend(b'RIFF')
    wav.extend(struct.pack('<I', 36 + data_size))
    wav.extend(b'WAVE')
    wav.extend(b'fmt ')
    wav.extend(struct.pack('<I', 16))
    wav.extend(struct.pack('<H', 1))
    wav.extend(struct.pack('<H', ch))
    wav.extend(struct.pack('<I', sample_rate))
    wav.extend(struct.pack('<I', sample_rate * ch * sw))
    wav.extend(struct.pack('<H', ch * sw))
    wav.extend(struct.pack('<H', sw * 8))
    wav.extend(b'data')
    wav.extend(struct.pack('<I', data_size))
    wav.extend(pcm)
    return bytes(wav)


def load_model():
    """Load VieNeuTTS model once at cold start."""
    global tts_model
    if tts_model is not None:
        return tts_model

    from vieneu import VieNeuTTS

    codec_device = os.getenv("CODEC_DEVICE", "cuda")
    backbone = os.getenv("BACKBONE_REPO", "pnnbao-ump/VieNeu-TTS-0.3B-q4-gguf")
    logger.info(f"Loading VieNeuTTS (backbone={backbone}, codec={codec_device})...")
    tts_model = VieNeuTTS(backbone_repo=backbone, codec_device=codec_device)
    logger.info("VieNeuTTS ready.")
    return tts_model


def handler(event):
    """RunPod serverless handler.

    Input (event["input"]):
      cmd: "infer" | "get_voice" | "encode_ref"

      For "infer":
        text: str - text to synthesize
        voice: str - preset voice name (e.g., "Doan")
        custom_voice: dict | null - custom voice codes (serialized)

      For "get_voice":
        voice: str - preset voice name

      For "encode_ref":
        audio_b64: str - base64-encoded reference audio
        ref_text: str - transcript of reference audio

    Output:
      For "infer":
        audio_b64: str - base64-encoded WAV audio
        sample_rate: int

      For "get_voice" / "encode_ref":
        voice: dict - serialized voice (tensors as lists)
    """
    import torch

    tts = load_model()
    inp = event.get("input", {})
    cmd = inp.get("cmd", "infer")

    if cmd == "infer":
        text = inp.get("text", "")
        voice_name = inp.get("voice", "Doan")
        custom_voice_data = inp.get("custom_voice")

        if not text:
            return {"error": "No text provided"}

        # Resolve voice
        if custom_voice_data:
            voice = {}
            for k, val in custom_voice_data.items():
                if isinstance(val, dict) and val.get("_tensor"):
                    voice[k] = torch.tensor(val["data"])
                else:
                    voice[k] = val
        else:
            voice = tts.get_preset_voice(voice_name)

        # Generate
        audio_np = tts.infer(text, voice=voice)
        wav_bytes = _numpy_to_wav(audio_np, tts.sample_rate)

        return {
            "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
            "sample_rate": tts.sample_rate,
        }

    elif cmd == "get_voice":
        voice_name = inp.get("voice", "Doan")
        voice = tts.get_preset_voice(voice_name)

        serialized = {}
        for k, val in voice.items():
            if isinstance(val, torch.Tensor):
                serialized[k] = {"_tensor": True, "data": val.cpu().tolist()}
            elif isinstance(val, np.ndarray):
                serialized[k] = {"_ndarray": True, "data": val.tolist()}
            else:
                serialized[k] = val

        return {"voice": serialized}

    elif cmd == "encode_ref":
        audio_b64 = inp.get("audio_b64", "")
        ref_text = inp.get("ref_text", "")

        if not audio_b64:
            return {"error": "No audio provided"}

        audio_bytes = base64.b64decode(audio_b64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name

        try:
            voice = tts.encode_reference(tmp_path, ref_text)
            serialized = {}
            for k, val in voice.items():
                if isinstance(val, torch.Tensor):
                    serialized[k] = {"_tensor": True, "data": val.cpu().tolist()}
                elif isinstance(val, np.ndarray):
                    serialized[k] = {"_ndarray": True, "data": val.tolist()}
                else:
                    serialized[k] = val
            return {"voice": serialized}
        finally:
            os.unlink(tmp_path)

    else:
        return {"error": f"Unknown command: {cmd}"}


# ── RunPod entrypoint ──────────────────────────────────────────
if __name__ == "__main__":
    import runpod

    # Pre-load model at cold start
    load_model()

    runpod.serverless.start({"handler": handler})
