#!/usr/bin/env bash
# One-time setup. Needs: Python 3.10+, ffmpeg, git, curl.
# Installs everything into ./.venv so it never touches your system Python.
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
[ -d .venv ] || "$PY" -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Optional: speaker-similarity check (narrate.py --check). resemblyzer's own webrtcvad
# dependency often fails to build, so install prebuilt wheels first. Narration works without it.
if python -m pip install webrtcvad-wheels librosa scipy && python -m pip install --no-deps resemblyzer; then
  echo "Similarity check available."
else
  echo "NOTE: similarity check unavailable (resemblyzer didn't install). Narration still works."
fi

mkdir -p models third_party
[ -d third_party/knn-vc ] || git clone https://github.com/bshall/knn-vc.git third_party/knn-vc
dl() { [ -s "models/$1" ] || curl -L --fail -o "models/$1" "$2"; }
dl kokoro-v1.0.onnx        https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
dl voices-v1.0.bin         https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
dl WavLM-Large.pt          https://github.com/bshall/knn-vc/releases/download/v0.1/WavLM-Large.pt
dl prematch_g_02500000.pt  https://github.com/bshall/knn-vc/releases/download/v0.1/prematch_g_02500000.pt

# Build the voice profile from your recording, if you've added one.
REF=$(ls voice_profile/reference.* 2>/dev/null | head -1 || true)
if [ -f voice_profile/voice_profile.pt ]; then
  echo "Voice profile already built."
elif [ -n "$REF" ]; then
  python make_voice_profile.py "$REF"
else
  echo
  echo "No voice recording found yet. Add a few minutes of clean speech as voice_profile/reference.wav"
  echo "(or .flac / .mp3 / .m4a), then run:  python make_voice_profile.py voice_profile/reference.wav"
fi
echo
echo "Setup complete. In each new terminal, first run:  source .venv/bin/activate"
