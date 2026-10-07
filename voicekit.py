"""Shared helpers for the voice pipeline (Kokoro TTS -> kNN-VC conversion into your voice).

kNN-VC: Baas, van Niekerk & Kamper, "Voice Conversion With Just Nearest Neighbors", Interspeech 2023.
Each 20 ms frame of the source speech (WavLM-Large layer-6 features) is replaced by the mean of its
k nearest frames from your voice profile, then a HiFi-GAN vocoder turns the features back into audio.
"""
import ctypes, gc, json, os, subprocess, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SR = 16000            # kNN-VC works at 16 kHz
WAVLM_LAYER = 6       # layer kNN-VC uses for speaker-dependent matching


def trim_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def load_audio(path, sr=SR):
    """Decode any audio/video file to mono float32 at `sr` with ffmpeg."""
    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def _knnvc_on_path(knnvc_dir):
    p = str(Path(knnvc_dir).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def load_wavlm(ckpt, knnvc_dir):
    _knnvc_on_path(knnvc_dir)
    from wavlm.WavLM import WavLM, WavLMConfig
    ck = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=False)
    model = WavLM(WavLMConfig(ck["cfg"]))
    model.load_state_dict(ck["model"])
    model.eval()
    del ck
    trim_memory()
    return model


def load_hifigan(ckpt, knnvc_dir):
    _knnvc_on_path(knnvc_dir)
    from hifigan.models import Generator
    from hifigan.utils import AttrDict
    with open(Path(knnvc_dir) / "hifigan" / "config_v1_wavlm.json") as f:
        cfg = AttrDict(json.load(f))
    gen = Generator(cfg)
    gen.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["generator"])
    gen.eval()
    gen.remove_weight_norm()
    return gen


@torch.inference_mode()
def wavlm_features(model, wav):
    """(T,) float32 16 kHz -> (frames, 1024) features, one frame per 20 ms."""
    x = torch.from_numpy(np.ascontiguousarray(wav))[None]
    return model.extract_features(x, output_layer=WAVLM_LAYER, ret_layer_results=False)[0].squeeze(0).clone()


def split_at_pauses(wav, target=7.0, window=2.0, sr=SR):
    """Cut points roughly every `target` s, placed at the quietest 50 ms within +/- `window` s."""
    pts, n, hop = [0], len(wav), int(0.05 * sr)
    while n - pts[-1] > (target + window) * sr:
        lo = pts[-1] + int((target - window) * sr)
        hi = min(n - sr, pts[-1] + int((target + window) * sr))
        best = min(range(lo, hi - hop, hop), key=lambda i: float(np.sqrt(np.mean(wav[i:i + hop] ** 2))))
        pts.append(best + hop // 2)
    pts.append(n)
    return pts


def load_profile(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    feats = obj["features"] if isinstance(obj, dict) else obj
    return feats.float()


@torch.inference_mode()
def convert_chunk(hifigan, feats, profile, profile_unit, k=4):
    """kNN-VC: replace each source frame by the mean of its k most similar profile frames, then vocode."""
    sims = F.normalize(feats, dim=-1) @ profile_unit.T
    idx = sims.topk(k=k, dim=-1).indices
    out = profile[idx].mean(dim=1)
    return hifigan(out[None]).squeeze().numpy()


def run(cmd):
    subprocess.run(cmd, shell=True, check=True)


def make_kokoro(model_path, voices_path):
    """Load Kokoro TTS.

    espeak-ng (Kokoro's pronunciation engine) can't read its data folder when the path is longer
    than about 160 characters, which deep install folders (e.g. inside iCloud Drive) can hit.
    In that case, copy the data once to a short temp folder and point Kokoro at it.
    """
    import os, shutil, tempfile
    from kokoro_onnx import Kokoro
    try:
        import espeakng_loader
        from kokoro_onnx import EspeakConfig
    except ImportError:
        return Kokoro(str(model_path), str(voices_path))
    data = str(espeakng_loader.get_data_path())
    if len(data) <= 120:
        return Kokoro(str(model_path), str(voices_path))
    try:
        from importlib.metadata import version
        tag = version("espeakng-loader")
    except Exception:
        tag = "x"
    short = os.path.join(tempfile.gettempdir(), f"lu-espeak-ng-data-{tag}")
    if not os.path.isfile(os.path.join(short, "phontab")):
        shutil.copytree(data, short, dirs_exist_ok=True)
    return Kokoro(str(model_path), str(voices_path), espeak_config=EspeakConfig(data_path=short))
