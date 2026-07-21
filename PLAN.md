# Speech-Transformer: Mic Testing → Server API → Mobile App (v1 Plan)

## Context

The Speech-Transformer repo can train an ASR model (mel spectrogram → encoder-decoder Transformer → text) but has **no inference path at all**: `translate.py` is a stale leftover from the text-translation ancestor project and doesn't work for speech. Needed, in order:

1. A way to test the trained model from the **desktop microphone** (training on LibriSpeech starts soon).
2. A **mobile app (iOS + Android)** that records meeting-length audio (up to 1+ hour), transcribes it via the model, lets the user **edit the transcript**, then **export/share** it (email etc.).

Decisions already made:

- **Inference runs on a server** (Python/FastAPI hosting the PyTorch model) — phone uploads audio, server returns transcript. Server = the GPU laptop first, cloud later.
- **React Native + Expo** for the app.
- **No backend storage or user accounts in v1** — transcripts are edited on-device and shared out; backend comes later if the product proves out.

Hard model constraint driving the design: the encoder handles max **500 mel frames = 16 s of audio** per pass (`max_audio_len` in config.py), so long recordings must be **chunked** and stitched.

## Architecture

One shared module — `inference.py` — is the single audio → transcript code path. The mic CLI and the FastAPI server are thin wrappers around it; the Expo app is a pure client of the server.

**Import rule:** `inference.py` imports only from `model.py`, `dataset.py`, `config.py`, `tokenizers` — never `train.py` (which drags in HF datasets/torchmetrics/tensorboard). Move `greedy_decode` (train.py:22-50) *into* `inference.py` and have `train.py` import it back, so there's one copy.

---

## Phase A — Inference core + desktop mic/file test

**Note:** this phase does not touch the model or training — the model is done. Today the repo has *no inference code at all* (`translate.py` is broken text-translation leftover). Phase A builds the audio→transcript engine (`inference.py`) that **Phase B's server directly reuses** — the only Phase-A-specific piece is a ~50-line CLI so the model can be tested from a file or the mic the day training finishes, before the server/app exist.

### New: `inference.py`

- `load_model_and_tokenizer(config, device, checkpoint_path=None)` — load `tokenizer_asr.json` first (vocab size drives model shape, mirroring `get_model()` train.py:145-153), `build_transformer(...)`, load `latest_weights_file_path(config)` checkpoint with `torch.load(..., map_location=device, weights_only=True)`, `model.eval()`. Keep tokenizer/checkpoint paths as parameters (the future SPGISpeech fine-tune has a different vocab → different projection-layer shape). **Gotcha:** `latest_weights_file_path` is cwd-relative (`Path('.')`) — resolve against the repo dir instead.
- Mel prep identical to `AudioDataset` (dataset.py:18-25, 66-91): MelSpectrogram(16000, n_fft=1024, hop=512, n_mels=80, slaney) + AmplitudeToDB → (time, n_mels) → zero-pad/truncate to 500 → mask.
- `greedy_decode` moved verbatim from train.py.
- `transcribe_waveform(model, tokenizer, waveform, sr, device, progress_cb=None)` — resample to 16 kHz mono, chunk (Phase B), decode per chunk, return `{"text", "segments": [{start, end, text}]}` under `torch.inference_mode()`.

### New: `transcribe_mic.py` (CLI, replaces translate.py)

- `--file clip.wav` mode first (test on LibriSpeech clips — flac/wav read via **soundfile**; avoid `torchaudio.load`, its I/O backends are in flux on Windows).
- `--mic [--seconds N]` mode with **sounddevice** (PortAudio wheels, works on Win11). **Gotcha:** some Windows drivers refuse 16 kHz capture — catch the error, record at device default (48 kHz) and resample down; print `sd.query_devices()` on failure and support `--device N`.

**Verify:** transcribe a LibriSpeech clip and compare to its reference; one-off assert that inference mel prep matches `AudioDataset.__getitem__` output; then mic-mode with a read-aloud LibriSpeech sentence.

## Phase B — Chunking + FastAPI server

### Chunking (in `inference.py`)

1. Decode whole file to 16 kHz mono float32 in RAM (1 h ≈ 230 MB — fine, no streaming in v1).
2. **silero-vad** for speech segments (energy-based splitter as `--vad energy` fallback).
3. Merge segments with <0.3 s gaps; greedily pack into chunks ≤ **15 s** (1 s headroom), cutting only at silence — so no word-splitting and no overlap logic needed.
4. Continuous speech >15 s with no gap: hard-split at the lowest-energy frame in the 12–15 s region. (Overlap-merge dedup explicitly deferred.)
5. Stitch chunk texts in order with timestamps.

Throughput on the RTX PRO 3000: no KV cache → ~1–3 s per 15 s chunk → **a 1-hour meeting takes ~5–15 min**. This is why the API must be job-based, not synchronous.

### New: `server.py` (FastAPI + uvicorn)

- `POST /v1/transcriptions` — multipart upload → save to `server_jobs/{job_id}/`, enqueue, return `job_id` immediately.
- `GET /v1/transcriptions/{job_id}` — `{status, chunks_done, chunks_total, text (partial, grows via progress_cb), error}`.
- `POST /v1/transcriptions/sync` — short clips only (≤60 s), for curl testing.
- `GET /health` — model/checkpoint info, used for LAN connectivity checks.
- Concurrency: `ThreadPoolExecutor(max_workers=1)` (GPU serialized; jobs queue). Job state = locked dict; job dirs TTL-cleaned (~24 h). **No DB.**
- **Audio formats:** Expo records AAC/.m4a; soundfile can't read it. Install ffmpeg once (`winget install Gyan.FFmpeg`) and pipe *every* upload through `ffmpeg -i in -ac 1 -ar 16000 -f wav` → soundfile. One decode path for all formats.
- CORS `allow_origins=["*"]` (one line; needed for browser-based testing). Serve `--host 0.0.0.0 --port 8000`; allow through Windows Firewall (private networks); phone on same Wi-Fi (corporate isolation → phone-hotspot fallback). Stub an `X-API-Key` check as a no-op now; real auth + HTTPS when moving to cloud.

**Verify:** concatenate LibriSpeech clips into a 2–3 min wav → check chunk ordering/timestamps via `--file`; curl the sync endpoint; curl the job flow and watch `chunks_done` advance; open `http://<laptop-ip>:8000/health` from the phone browser.

## Phase C — Expo app (`mobile/` directory, `npx create-expo-app`, SDK ≥53)

Modules:

- **expo-audio** for recording (current API; `expo-av` audio is deprecated). AAC m4a ~64 kbps mono → 1 h ≈ 30 MB. Don't force 16 kHz on device; server resamples.
- **expo-file-system** for upload with progress — **gotcha:** on SDK 54+ the `createUploadTask` progress API lives under `expo-file-system/legacy`.
- Share: RN `Share.share({message})` for text + `expo-sharing` with a written `.txt` for email attachments.
- **AsyncStorage** for drafts `{jobId, createdAt, title, editedText, status}`, debounce-saved on every edit. This is the entire v1 storage layer.
- `expo-keep-awake` while recording; `expo-router` for navigation.

Screens: **Record** (button, timer, keep-awake, drafts list) → **Processing** (upload progress → poll job every 2 s, show chunks progress + growing partial transcript; persist jobId immediately so a killed app resumes polling) → **Transcript** (multiline TextInput pre-filled, autosave, Share text / Share .txt / Copy) → minimal **Settings** (server URL field, default `http://<laptop-ip>:8000`).

Platform gotchas: v1 requires the app foregrounded while recording (true background recording — iOS `UIBackgroundModes: audio`, Android foreground service — deferred); plain `http://` works in Expo Go dev but production builds need cleartext/ATS exceptions or an HTTPS server; accept iOS local-network prompt; on phone-call interruption save the partial recording.

**Verify:** 10 s record → upload → poll → edit → share to email; then a 30–60 min real test (record a podcast playing out loud), including kill-app-mid-poll → reopen → resume.

---

## Build order (gates)

| # | Do | Gate |
|---|---|---|
| 1 | inference.py + file mode | LibriSpeech clip transcribes; mel matches AudioDataset |
| 2 | Mic mode | Read-aloud sentence → plausible output |
| 3 | VAD chunking | 2–3 min wav → ordered timestamped segments |
| 4 | server sync endpoint | curl wav → transcript |
| 5 | Job endpoints | poll shows progress + partial text |
| 6 | LAN | Phone browser hits /health |
| 7 | Expo happy path | record → transcript → edit → share |
| 8 | Long-recording hardening | 30–60 min end-to-end + kill/resume |

New deps: `sounddevice soundfile silero-vad fastapi uvicorn[standard] python-multipart` (+ ffmpeg via winget).

## Expectations note

LibriSpeech-trained, word-level tokenizer, greedy decoding → rough transcripts on real meetings (uppercase, no punctuation, `[UNK]` for finance jargon). That's expected; the pipeline being built is model-agnostic, and the planned SPGISpeech fine-tune drops into the same `inference.py` later.

## Explicitly deferred (v2+)

Accounts/backend DB · on-device inference (ONNX/ExecuTorch) · live/streaming transcription · speaker diarization · punctuation/casing restoration · beam search / batched decode / KV cache · Android background recording · HTTPS+auth · overlap-merge chunking. Delete `translate.py` once transcribe_mic.py works.

## Critical files

- `train.py` (greedy_decode source, lines 22-50; model construction pattern lines 145-153)
- `dataset.py` (exact mel/pad/mask pipeline, lines 18-25, 66-91)
- `config.py` (hyperparams; cwd-relative path gotcha in `latest_weights_file_path`)
- New: `inference.py`, `transcribe_mic.py`, `server.py`, `mobile/` (Expo app)
