# Checkpoint — 2026-07-21: Inference core + server scaffolding (pre-training)

Status snapshot after implementing PLAN.md Phase A and Phase B scaffolding.
**No trained weights exist yet** — everything below was built and verified so
that transcription is a one-command test the moment training produces
`tokenizer_asr.json` + a checkpoint in `librispeech_weights/`.

## What was built

| File | Role |
|------|------|
| `inference.py` | **New.** The single ASR engine: model/tokenizer loading (repo-relative paths, random-init fallback when no weights exist), mel prep identical to `AudioDataset`, `greedy_decode` (moved here from train.py), VAD chunking with fixed/overlap fallback, `transcribe_waveform` / `transcribe_file` |
| `transcribe_mic.py` | **New.** Desktop CLI: `--file`, `--mic`, `--segments`, `--vad silero|energy|fixed`, `--checkpoint`, `--tokenizer`, `--list-devices` |
| `test_mel_parity.py` | **New.** Gate A2 check: inference mel prep vs `AudioDataset.__getitem__` |
| `server.py` | **New.** FastAPI: `GET /health`, `POST /v1/transcriptions` (job) + `GET /v1/transcriptions/{id}` (poll, growing partial text), `POST /v1/transcriptions/sync` (≤60 s), single GPU worker queue, ffmpeg upload conversion (soundfile fallback for wav/flac/ogg), no-op `X-API-Key` stub, 24 h job TTL cleanup |
| `train.py` | **Changed.** `greedy_decode` removed; imports it back from `inference.py` (single-copy rule) |
| `translate.py` | **Quarantined.** Stale translation-era script now exits with a pointer to `transcribe_mic.py`; delete after Gate A1 passes with real weights |
| `requirements.txt` | Added: `soundfile`, `sounddevice`, `silero-vad`, `fastapi`, `uvicorn[standard]`, `python-multipart` |
| `README.md` | Added "Test transcription (after training finishes)" + server sections |
| `.gitignore` | Added `server_jobs/` |

## Verification (this machine: RTX PRO 3000, torch 2.13.0+cu130 — random weights + dummy tokenizer)

- `smoke_test_offline.py` — **PASSED** (after the train.py refactor)
- Mel parity (Gate A2) — **PASSED, bit-exact** (`max_diff = 0.0`) on pad, ~boundary, and truncate paths
- `transcribe_mic.py --file` short clip — runs end-to-end on **cuda**, single pass, prints (garbage) transcript
- `--file` 150 s clip, energy VAD — 15 ordered, timestamped chunks (Gate B1 at pipeline level)
- `--file` 150 s clip, silero VAD — silero found no speech in synthetic noise → **fixed 15 s windows + 1.5 s overlap fallback engaged automatically** (Gate B2)
- Server `/health` — OK (reports device, checkpoint, tokenizer, ffmpeg, active jobs)
- Server sync — transcribed short clip; correctly rejected 150 s clip with **HTTP 413**
- Server job flow — upload → `job_id` → polls showed `chunks_done` 1→12 with growing partial text → `done` (Gates B3/B4 at pipeline level)
- `import train` — OK; `greedy_decode.__module__ == 'inference'`

## Gates pending prerequisites (not skipped)

| Gate | Blocked on |
|------|-----------|
| A1 — real file transcription vs LibriSpeech reference | trained checkpoint + tokenizer |
| A3 — live mic transcription | trained checkpoint; also run `--mic` once to confirm capture on this hardware (code-complete, not yet live-tested) |
| B5 — phone opens `http://<laptop-ip>:8000/health` on LAN | phone on same Wi-Fi; server side ready (`--host 0.0.0.0`) |

## Commands to run after training finishes

```powershell
python test_mel_parity.py                                # sanity
python transcribe_mic.py --file path\to\librispeech.flac # Gate A1
python transcribe_mic.py --mic --seconds 10              # Gate A3
uvicorn server:app --host 0.0.0.0 --port 8000            # server
curl -F "file=@clip.wav" http://localhost:8000/v1/transcriptions/sync
```

Checkpoints and `tokenizer_asr.json` are discovered automatically (latest
checkpoint wins); `--checkpoint` / `--tokenizer` override.

## Known items / next

- **ffmpeg not installed** on this laptop yet — wav/flac uploads work without
  it; phone m4a uploads need `winget install Gyan.FFmpeg`.
- `train_wb.py` still contains a stale translation-era `greedy_decode` copy
  (leftover, unused by the new code) — clean up or delete later.
- Next per PLAN.md: finish Lifetime 0 (train on LibriSpeech), pass Gates
  A1/A3 with real weights, then Phase C (Expo app).
