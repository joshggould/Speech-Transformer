# Speech Transformer

Mel spectrogram → Transformer encoder → text decoder (ASR), adapted from a from-scratch translation Transformer.

## Setup

```bash
git clone https://github.com/joshggould/Speech-Transformer.git
cd Speech-Transformer
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# On NVIDIA: install CUDA build of torch/torchaudio from pytorch.org
```

## Quick smoke test (no dataset download)

```bash
python smoke_test_offline.py
```

## Train (LibriSpeech clean train.100)

```bash
python train.py
```

## Test transcription (after training finishes)

Training must have produced `tokenizer_asr.json` (repo root) and at least one
checkpoint in `librispeech_weights/smodel_XX.pt`. Both are picked up
automatically — no flags needed:

```bash
# 1. Sanity: mel pipeline still matches training exactly
python test_mel_parity.py

# 2. Transcribe an audio file (wav/flac); best first test = a LibriSpeech clip
python transcribe_mic.py --file path/to/clip.flac

# 3. Transcribe from the microphone (10 s; see --list-devices if capture fails)
python transcribe_mic.py --mic --seconds 10

# Long recordings are chunked automatically; see per-chunk timestamps:
python transcribe_mic.py --file meeting.wav --segments

# Use a specific checkpoint instead of the latest:
python transcribe_mic.py --file clip.wav --checkpoint librispeech_weights/smodel_05.pt
```

Before weights exist, all of the above still run with a random-init model and
dummy tokenizer (garbage text, pipeline check only) — a loud WARNING says so.

## Transcription server (for the mobile app)

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

- `GET /health` — model/device info (also the phone's connectivity check)
- `POST /v1/transcriptions` — upload audio, returns `{job_id}` immediately
- `GET /v1/transcriptions/{job_id}` — poll: status, chunk progress, growing text
- `POST /v1/transcriptions/sync` — short clips (≤60 s) inline, for curl tests

```bash
curl -F "file=@clip.wav" http://localhost:8000/v1/transcriptions/sync
```

Phone uploads (AAC/m4a) need ffmpeg on the server:
`winget install Gyan.FFmpeg.Shared` (the *shared* build — training also needs
its DLLs for torchcodec, which `datasets` uses to decode LibriSpeech audio).
Without it only wav/flac/ogg uploads work. Open a fresh terminal after
installing so PATH picks it up.

## Layout

| File | Role |
|------|------|
| `dataset.py` | `AudioDataset` — mel encoder input + transcript decoder |
| `model.py` | Transformer with `Linear(n_mels → d_model)` on encoder |
| `train.py` | LibriSpeech training loop |
| `config.py` | Hyperparameters |

Translation reference project is kept separate; this repo is speech-only.

## Planned next

1. Train general ASR on LibriSpeech (NVIDIA)
2. Fine-tune on financial speech (e.g. SPGISpeech)
3. Evaluate on Earnings-21 / Earnings-22
