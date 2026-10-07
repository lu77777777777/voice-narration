# Voice Narration Pipeline

Turn a written script into narration **in your own voice**, from a few minutes of your recorded speech. It's open source and runs locally.

```
script.json ──► Kokoro-82M ──► generic voice ──► kNN-VC ──► narration in your voice
                (text-to-speech)                 (voice conversion,
                                                  using your voice profile)
```

## How it works

1. **Pronunciation.** Each line of the script goes through `pronunciations.json`, so technical terms are said correctly (for example "RAG" → "rag", "LLM" → "L L M", "BM25" → "B M twenty-five").
2. **Text-to-speech.** [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) reads the script in a generic voice (`af_heart` by default).
3. **Voice conversion.** [kNN-VC](https://github.com/bshall/knn-vc) (Baas, van Niekerk & Kamper, Interspeech 2023) replaces the generic voice with yours:
   - WavLM-Large cuts the speech into 20 ms frames.
   - Each frame is replaced by the average of its 4 nearest frames from **your voice profile**, which is built from your own recording.
   - A HiFi-GAN vocoder turns the result back into audio.

The words and timing stay exactly as Kokoro read them; only the voice changes.

## Results and limitations

From my own use (about 3.8 minutes of reference audio):

- **Speaker similarity:** 0.87–0.91 (Resemblyzer). For comparison, the same speaker scored against themselves gets about 0.99.
- **Intelligibility:** a Whisper speech recogniser transcribes the output almost word for word, technical terms included.
- **Delivery:** this is the main limitation. Rhythm, emphasis and intonation come from the generic text-to-speech voice, not from you, so the narration can sound flat or robotic. Viewers of my first video noticed this.
- **Audio quality:** output is 16 kHz, slightly duller than a real recording.

**Listen before you publish.** The similarity score tells you whether it sounds like *you*, not whether it sounds *human*.

## Quick start

You need Python 3.10+, ffmpeg, git and curl.

```bash
git clone https://github.com/lu77777777777/voice-narration.git && cd voice-narration
# 1. Add a few minutes of clean speech in your own voice (see voice_profile/README.md)
cp ~/my_recording.wav voice_profile/reference.wav
# 2. One-time setup: private Python environment, ~1.6 GB of models, your voice profile
./setup.sh
# 3. Narrate the example and score the voice match
source .venv/bin/activate
python narrate.py examples/quickstart.json --out narration/quickstart --check voice_profile/reference.wav
```

The audio files are written to `narration/quickstart/voice/<id>.wav`.

In every new terminal, run `source .venv/bin/activate` first.

## Usage

**A script** is a JSON list with one item per audio file, in order. For a video, that's one item per slide:

```json
[
  {"id": "intro", "text": "Hi everyone, and welcome back. Today we're looking at how RAG works."},
  {"id": "detail", "text": "Hybrid search combines BM25 keyword matching with vector search."}
]
```

Write terms normally ("RAG", "LLM"). To change how a term is said, add a rule to `pronunciations.json`.

**Fix a single item** after editing its text:

```bash
python narrate.py script.json --out narration/myvideo --only detail
```

Finished items are reused, so a rerun only generates what's missing.

**Options:**

| Option | Default | What it does |
|---|---|---|
| `--check REF_AUDIO` | off | Scores each output's speaker similarity against your recording (0.87–0.91 is typical) |
| `--k` | 4 | How many of your voice frames are averaged per 20 ms. Lower (e.g. 2) sounds less smoothed but may glitch more |
| `--voice` | `af_heart` | Kokoro voice used for the delivery |
| `--speed` | 1.0 | Kokoro speaking speed (e.g. 0.92 is a little slower and calmer) |
| `--pron none` | | Skip the pronunciation rules |

**Improving the voice:** more of your voice in the profile helps most. Record 10 or more minutes of clean narration, then add it:

```bash
python make_voice_profile.py my_new_recording.wav --append
```

A GPU is optional. On one CPU core, narration takes about twice the length of the audio. 4 GB of RAM is enough, because the audio is processed in short chunks.

## Troubleshooting

- **`Error processing file '.../espeak-ng-data/phontab'`.** Kokoro's pronunciation engine can't read its data from folder paths longer than about 160 characters. `narrate.py` handles this automatically by copying the data to a short temporary folder. If you still see it, move the repo to a shorter path.
- **"similarity check unavailable" during setup.** The optional checker didn't install. Narration still works; only `--check` is unavailable.
- **`ModuleNotFoundError`.** The environment isn't active. Run `source .venv/bin/activate`.

## Responsible use

This is voice cloning.

- **Only clone your own voice, or a voice whose owner has given explicit consent.**
- **Don't use it to impersonate anyone or to mislead listeners.** Synthetic voices are a real fraud risk.
- **Follow your platform's rules on synthetic content** when you publish.
- **Keep your recordings private.** `.gitignore` keeps everything in `voice_profile/` out of git; anyone with your recording can clone your voice.

## Credits and licences

This repo is MIT-licensed (see `LICENSE`). The upstream components are not included in it; `setup.sh` downloads them:

| Component | Licence |
|---|---|
| [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) via [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) | Apache-2.0 / MIT |
| [kNN-VC](https://github.com/bshall/knn-vc), including its WavLM-Large and HiFi-GAN checkpoints | MIT |
| [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) (optional check) | Apache-2.0 |

## Tested

Tested on 7 October 2026 from a fresh copy (Linux, Python 3.13):

- **Setup:** completes with and without a voice recording present.
- **Profile build:** works from the recording.
- **Narration and `--check`:** the example script narrates and scores 0.87–0.89.
- **Intelligibility:** a Whisper speech recogniser transcribes the output almost word for word (one slip: "reranker" heard as "reranger").
- **Cache and `--only`:** a rerun reuses finished files, and `--only` regenerates just the item named.

Not yet tested on macOS or Windows.
