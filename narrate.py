"""Narrate a script in your voice.

  python narrate.py examples/quickstart.json --out narration/quickstart
  python narrate.py script.json --out narration/x --only cover hook     # redo selected slides

Script = JSON list of {"id": "...", "text": "..."}, one item per audio file (e.g. per slide), in order.
Before speaking, each text goes through pronunciations.json (e.g. "RAG" -> "rag", "LLM" -> "L L M");
edit that file to fix how a term is said.
Pipeline per slide:
  1. Kokoro-82M TTS (voice af_heart: a clear, neutral female voice; try others with --voice) -> tts/<id>.wav (24 kHz)
  2. WavLM-Large layer-6 features, in ~7 s chunks cut at pauses   (WavLM is freed before step 3)
  3. kNN-VC (k=4) against your voice profile + HiFi-GAN vocoder   -> voice/<id>.wav (16 kHz)
Existing outputs are reused; delete a file (or use --only) to regenerate it.
"""
import argparse, json, re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import voicekit as vk

ap = argparse.ArgumentParser()
ap.add_argument("script")
ap.add_argument("--out", required=True)
ap.add_argument("--profile", default="voice_profile/voice_profile.pt")
ap.add_argument("--models", default="models")
ap.add_argument("--knnvc", default="third_party/knn-vc")
ap.add_argument("--voice", default="af_heart")
ap.add_argument("--speed", type=float, default=1.0)
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--pron", default=str(Path(__file__).with_name("pronunciations.json")),
                help="pronunciation rules applied before speaking ('none' to skip)")
ap.add_argument("--only", nargs="*", help="slide ids to (re)generate")
ap.add_argument("--check", metavar="REF_AUDIO", help="score speaker similarity vs this recording (needs resemblyzer)")
a = ap.parse_args()

items = json.load(open(a.script))
rules = json.load(open(a.pron))["rules"] if a.pron != "none" and Path(a.pron).exists() else []
def spoken(text):
    for pat, rep in rules:
        text = re.sub(pat, rep, text)
    return re.sub(r"\s+", " ", text).strip()
if a.only:
    items = [it for it in items if it["id"] in a.only]
out = Path(a.out)
for d in ("tts", "feats", "voice"):
    (out / d).mkdir(parents=True, exist_ok=True)
if a.only:
    for it in items:
        for p in (out / "tts" / f"{it['id']}.wav", out / "feats" / f"{it['id']}.pt", out / "voice" / f"{it['id']}.wav"):
            p.unlink(missing_ok=True)

# 1) TTS
todo = [it for it in items if not (out / "tts" / f"{it['id']}.wav").exists()]
if todo:
    kokoro = vk.make_kokoro(Path(a.models) / "kokoro-v1.0.onnx", Path(a.models) / "voices-v1.0.bin")
    for it in todo:
        audio, sr = kokoro.create(spoken(it["text"]), voice=a.voice, speed=a.speed, lang="en-us")
        sf.write(out / "tts" / f"{it['id']}.wav", audio, sr)
        print(f"[tts] {it['id']}: {len(audio) / sr:.1f}s")
    del kokoro

# 2) WavLM features
todo = [it for it in items if not (out / "feats" / f"{it['id']}.pt").exists()]
if todo:
    wavlm = vk.load_wavlm(Path(a.models) / "WavLM-Large.pt", a.knnvc)
    for it in todo:
        wav = vk.load_audio(out / "tts" / f"{it['id']}.wav")
        pts = vk.split_at_pauses(wav)
        chunks = []
        for s, e in zip(pts[:-1], pts[1:]):
            chunks.append((vk.wavlm_features(wavlm, wav[s:e]), e - s))
            vk.trim_memory()
        torch.save(chunks, out / "feats" / f"{it['id']}.pt")
        print(f"[features] {it['id']}: {len(chunks)} chunks")
    del wavlm
    vk.trim_memory()

# 3) Convert to your voice
todo = [it for it in items if not (out / "voice" / f"{it['id']}.wav").exists()]
if todo:
    hifigan = vk.load_hifigan(Path(a.models) / "prematch_g_02500000.pt", a.knnvc)
    profile = vk.load_profile(a.profile)
    unit = torch.nn.functional.normalize(profile, dim=-1)
    for it in todo:
        pieces = []
        for feats, n in torch.load(out / "feats" / f"{it['id']}.pt", weights_only=False):
            w = vk.convert_chunk(hifigan, feats, profile, unit, k=a.k)
            pieces.append(np.pad(w, (0, max(0, n - len(w))))[:n])
            vk.trim_memory()
        y = np.concatenate(pieces).astype(np.float32)
        sf.write(out / "voice" / f"{it['id']}.wav", y, vk.SR)
        print(f"[voice] {it['id']}: {len(y) / vk.SR:.1f}s")

# Optional: speaker similarity check (same speaker vs itself ~0.99; this pipeline typically scores ~0.89)
if a.check:
    from resemblyzer import VoiceEncoder, preprocess_wav
    enc = VoiceEncoder("cpu")
    ref = enc.embed_utterance(preprocess_wav(vk.load_audio(a.check), source_sr=vk.SR))
    for it in items:
        e = enc.embed_utterance(preprocess_wav(vk.load_audio(out / "voice" / f"{it['id']}.wav"), source_sr=vk.SR))
        print(f"[similarity] {it['id']}: {float(np.dot(e, ref) / np.linalg.norm(e) / np.linalg.norm(ref)):.3f}")
