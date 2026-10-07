"""Build (or extend) your voice profile: WavLM features of your real speech, used as the kNN-VC matching set.

  python make_voice_profile.py voice_profile/reference.wav
  python make_voice_profile.py new_recording.wav --append      # add more of your voice later

More clean audio of your voice = better conversion. 3-4 min works; 10+ min is noticeably better.
Use clean narration only (no music, no other speakers).
"""
import argparse, datetime
from pathlib import Path

import torch

import voicekit as vk

ap = argparse.ArgumentParser()
ap.add_argument("audio", nargs="+", help="recordings of your voice (wav/flac/mp3/m4a/mp4...)")
ap.add_argument("--out", default="voice_profile/voice_profile.pt")
ap.add_argument("--append", action="store_true", help="add to an existing profile instead of replacing it")
ap.add_argument("--models", default="models")
ap.add_argument("--knnvc", default="third_party/knn-vc")
a = ap.parse_args()

torch.set_num_threads(max(1, torch.get_num_threads()))
wavlm = vk.load_wavlm(Path(a.models) / "WavLM-Large.pt", a.knnvc)
chunks, sources = [], []
for f in a.audio:
    wav = vk.load_audio(f)
    step = 10 * vk.SR                      # 10 s pieces keep memory low
    for i in range(0, len(wav), step):
        piece = wav[i:i + step]
        if len(piece) > vk.SR // 2:
            chunks.append(vk.wavlm_features(wavlm, piece))
            vk.trim_memory()
    sources.append({"file": Path(f).name, "seconds": round(len(wav) / vk.SR, 1)})
    print(f"{f}: {len(wav) / vk.SR:.1f}s")

feats = torch.cat(chunks, dim=0)
out = Path(a.out)
if a.append and out.exists():
    old = torch.load(out, map_location="cpu", weights_only=False)
    feats = torch.cat([vk.load_profile(out), feats], dim=0)
    sources = (old.get("sources", []) if isinstance(old, dict) else []) + sources
out.parent.mkdir(parents=True, exist_ok=True)
torch.save({"features": feats, "layer": vk.WAVLM_LAYER, "sr": vk.SR, "sources": sources,
            "created": datetime.datetime.now().isoformat(timespec="seconds")}, out)
print(f"Saved {out}: {feats.shape[0]} frames (~{feats.shape[0] * 0.02 / 60:.1f} min of your voice)")
