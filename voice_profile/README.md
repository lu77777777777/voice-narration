# Your voice goes here (it is never committed)

Put a recording of **your own voice** here as `reference.wav` (`.flac`, `.mp3` or `.m4a` also work):

- 3–4 minutes of clean narration works; 10+ minutes is noticeably better.
- A quiet room, no music, no other speakers, the same mic you'll keep using.
- Read in your normal presenting voice.

`setup.sh` (or `python make_voice_profile.py voice_profile/reference.wav`) turns it into `voice_profile.pt`.

Everything in this folder except this README is ignored by git, so your recording and profile stay on your machine. Keep a backup of the recording: it's what makes the pipeline sound like you.
