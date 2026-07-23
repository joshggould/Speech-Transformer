# Speech-Transformer — Full Project Lifetime Plan

This document is the **source of truth** for how the project should evolve from “trained ASR model” → “desktop test” → “GPU server API” → “mobile meeting app” → “financial-domain ASR.”

Feed this whole file to coding agents. Prefer implementing **one gated step at a time**; do not skip gates.

---

## 0. One-sentence product

Record meeting-length audio (up to 1+ hour) on a phone, transcribe it with **our** Speech Transformer on a GPU server, let the user **edit** the transcript on-device, then **export/share** (email, etc.) — with a later fine-tune for **financial / earnings-call** language (EBITDA, etc.).

---

## 1. Current repo state (what exists today)

| Piece | Status |
|-------|--------|
| `model.py` | Speech Transformer: mel → `nn.Linear(n_mels → d_model)` → encoder → decoder → text |
| `dataset.py` | `AudioDataset`: waveform → Mel+dB → `(T, n_mels)` pad/truncate to `max_audio_len` + text decoder labels |
| `train.py` | LibriSpeech `clean` / `train.100` training loop, WER/CER validation |
| `config.py` | `n_mels=80`, `max_audio_len=500`, `seq_len=350`, **BPE** tokenizer (`tokenizer_asr_bpe.json`, vocab ~4000) |
| `smoke_test_offline.py` | Shape smoke test (no HF download) |
| `translate.py` | **Stale** text-translation leftover — does **not** work for speech; delete after Phase A |
| Inference / mic / API / mobile | **Do not exist yet** |

### Hard model constraint (drives all long-audio design)

- Encoder max **500 mel frames**.
- Mel hop = 512 samples @ 16 kHz → **≈ 16 seconds** of audio per forward pass.
- **Do not** raise `max_audio_len` to cover 1 hour. Attention is O(T²); ~1 hour ≈ ~112k frames — not viable.
- Long audio = **chunk → decode each chunk → stitch transcripts**.

### Tokenizer constraint (drives finance quality later)

- **Now (retrain path):** **BPE** (~4000 merges) → `tokenizer_asr_bpe.json`, weights under `librispeech_bpe_weights/`.
- Old WordLevel run (`tokenizer_asr.json` / `librispeech_weights/`) is a baseline only — do **not** preload those into a BPE model.
- BPE can form rare/finance words from pieces; still rebuild/extend BPE on Libri+SPGISpeech text before finance fine-tune.
- Always ship **checkpoint + matching tokenizer** together (vocab size sets the projection layer).

### Quality expectations (v1)

LibriSpeech-trained + BPE + greedy decode → better than WordLevel, still rough on real meetings (limited punctuation, domain mismatch). **That is expected.** The pipeline must be model-agnostic so a better checkpoint/tokenizer drops in later without rewriting the app.

---

## 2. Architecture (end state)

```
┌─────────────────┐     HTTPS/HTTP LAN      ┌──────────────────────────────┐
│ Expo app (phone)│ ──────────────────────► │ FastAPI server (GPU laptop)  │
│ record / edit / │ ◄────────────────────── │  load model once             │
│ share           │   job status + text     │  inference.py                │
└─────────────────┘                         │  chunk long audio            │
                                            │  queue GPU jobs              │
                                            └──────────────────────────────┘
```

**Single shared core:** `inference.py` = the only audio→transcript implementation.

| Layer | Role |
|-------|------|
| `inference.py` | load model/tokenizer, mel prep (= dataset), chunking, greedy decode, `transcribe_*` |
| `transcribe_mic.py` | CLI: `--file` / `--mic` → calls inference |
| `server.py` | FastAPI: upload → job queue → poll transcript |
| `mobile/` | Expo client only — never runs PyTorch |

**Import rule:** `inference.py` may import `model.py`, `dataset.py` (transforms/helpers only if needed), `config.py`, `tokenizers`.  
**Never** import `train.py` from inference/server (pulls datasets/torchmetrics/tensorboard).

**Decode ownership:** Move `greedy_decode` from `train.py` into `inference.py`. Have `train.py` import it back so there is **one** copy.

---

## 3. Full project lifetime (phases over time)

Think of the project as **epochs of the product**, not just engineering tickets.

```
Lifetime 0  Learn + train general ASR (LibriSpeech)          [NOW / NVIDIA]
Lifetime 1  Inference core + desktop proof                  [Phase A]
Lifetime 2  Long-form chunking + GPU server                 [Phase B]
Lifetime 3  Mobile record → edit → share                     [Phase C]
Lifetime 4  Domain: finance tokenizer + SPGISpeech fine-tune
Lifetime 5  Evaluate on Earnings-21/22; iterate model
Lifetime 6  Harden product (auth, HTTPS, cloud, polish)     [v2+]
```

Do not start Lifetime 3 UI polish before Lifetime 1 has a checkpoint that transcribes a known LibriSpeech clip. Pipeline scaffolding can be coded in parallel, but **gates require real weights**.

---

## Lifetime 0 — Train general English ASR (NVIDIA)

**Goal:** A checkpoint that maps speech → text on clean read speech.

### Commands (NVIDIA machine)

```bash
git clone https://github.com/joshggould/Speech-Transformer.git
cd Speech-Transformer
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# If CUDA not visible, install CUDA torch/torchaudio from pytorch.org
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
python smoke_test_offline.py
python train.py
```

Expect: `Using device: cuda`, download of LibriSpeech `train.100`, weights under `librispeech_weights/` (or config `datasource` + `model_folder`).

### Gate 0 (must pass before relying on mic/app demos)

- [ ] Training runs without shape errors  
- [ ] Validation prints TARGET vs PREDICTED that are at least vaguely related  
- [ ] At least one `.pt` checkpoint saved  
- [ ] `tokenizer_asr_bpe.json` exists next to training cwd  
- [ ] At least one `.pt` under `librispeech_bpe_weights/`  

**Do not** fine-tune finance yet. **Do not** block Lifetime 1 on perfect WER.

---

## Lifetime 1 / Phase A — Inference core + desktop mic/file test

**Goal:** Same mel path as training → transcript from a file or mic. No mobile yet.

### Why this phase exists

Today there is **no speech inference path**. `translate.py` is broken leftover. Phase A builds the engine Phase B’s server will call.

### New file: `inference.py`

Implement at least:

#### `load_model_and_tokenizer(config, device, checkpoint_path=None, tokenizer_path=None)`

1. Resolve paths **against the repo directory**, not bare `Path('.')` (fix cwd-relative bug in `latest_weights_file_path` / `get_weights_file_path`).
2. Load tokenizer **first** (`tokenizer_asr.json` or override) — vocab size drives `build_transformer(..., tgt_vocab_size=...)`.
3. `build_transformer(n_mels, vocab, max_audio_len, seq_len, d_model=...)`.
4. `torch.load(checkpoint, map_location=device, weights_only=True)` → `model.load_state_dict(...)` → `model.eval()`.
5. Return `(model, tokenizer, config)`.

**Hard rule:** Always load a tokenizer that matches the checkpoint. Finance fine-tune later = different vocab = different projection weights. Pass both paths explicitly when not using defaults.

#### Mel prep (must match `AudioDataset`)

Identical to training:

- Resample to **16 kHz**, mono, float32  
- `MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=512, n_mels=80, mel_scale="slaney")`  
- `AmplitudeToDB()`  
- Squeeze channel → transpose to `(time, n_mels)`  
- Truncate/pad time to `max_audio_len` (500)  
- `encoder_mask`: 1 for real frames, 0 for pad; shape `(1, 1, max_audio_len)`  

**Unit check:** For the same waveform, inference mel/`encoder_input` must match `AudioDataset.__getitem__` (atol reasonable for float).

#### `greedy_decode` 

Move verbatim from `train.py` (keep decoder ids as `torch.long`, not `type_as(mel)`).

#### `transcribe_waveform(model, tokenizer, waveform, sr, device, chunking=None, progress_cb=None)`

- Resample/mono  
- If duration ≤ ~15–16 s: single encode/decode  
- If longer: use chunking module (Lifetime 2) — Phase A may require `--file` short clips only until chunking lands  
- Under `torch.inference_mode()`  
- Return:

```python
{
  "text": "FULL STITCHED TRANSCRIPT",
  "segments": [
    {"start": 0.0, "end": 14.2, "text": "..."},
    ...
  ]
}
```

### New file: `transcribe_mic.py` (CLI; replaces `translate.py`)

**Deps:** `soundfile`, `sounddevice` (add to requirements).

Modes:

1. `--file path.wav|.flac` — prefer **soundfile** for I/O on Windows (torchaudio backends are flaky across platforms).  
2. `--mic [--seconds N] [--device N]` — PortAudio via sounddevice.

**Windows mic gotcha:** Some drivers refuse 16 kHz capture. Catch error → record at device default (often 48 kHz) → resample to 16 kHz. On failure print `sd.query_devices()` and support `--device N`.

### Phase A gates

| # | Gate | Pass criteria |
|---|------|----------------|
| A1 | File transcription | LibriSpeech clip → text; eyeball vs reference |
| A2 | Mel parity | Inference mel matches dataset mel for same audio |
| A3 | Mic mode | Read aloud a known sentence → plausible transcript |

**After A1–A3:** Delete or quarantine `translate.py` so nobody runs it by mistake.

---

## Lifetime 2 / Phase B — Long-form chunking + FastAPI server

**Goal:** 1+ hour audio file → transcript via jobs on the GPU laptop; phone can hit `/health` on LAN.

### 2.1 Chunking policy (IMPORTANT — revised)

**Primary (preferred):** silence-aware packing  

1. Decode whole file to 16 kHz mono float32 in RAM (1 h ≈ 230 MB — OK for v1; no streaming).  
2. Run **silero-vad** to find speech segments (`--vad silero`).  
3. Energy-based splitter as `--vad energy` fallback if silero unavailable.  
4. Merge segments with gaps **&lt; 0.3 s**.  
5. Greedily pack into chunks **≤ 15 s** (1 s headroom under 16 s model limit).  
6. Cut only at silence when possible → fewer mid-word splits; **no overlap merge needed** for those chunks.  
7. If one continuous speech region **&gt; 15 s** with no gap: hard-split at the **lowest-energy frame** in the 12–15 s region.

**Secondary (required safety net — do not ship without this):** fixed windows + overlap  

When VAD finds nothing useful, or audio is noisy / music / crosstalk (common on earnings calls and laptop mics):

1. Fall back to fixed windows (e.g. **15 s** window, **1–2 s** overlap).  
2. Decode each window.  
3. Stitch with simple overlap handling v1: prefer the non-overlap region; optional naive dedup of repeated trailing/leading words.  
4. Full clever overlap-merge can stay v2, but **fixed+overlap fallback must exist** so long jobs never hang on empty VAD.

**Config knobs (put in config or CLI):**

- `chunk_seconds` (default 15)  
- `chunk_overlap_seconds` (default 1.5) — used only in fallback mode  
- `vad_mode`: `silero` | `energy` | `fixed`  

**Throughput note (RTX-class laptop, no KV cache):** ~1–3 s compute per ~15 s audio → **1 hour meeting ≈ 5–15 minutes** wall time. Therefore the API **must be job-based**, not a single blocking HTTP request for long files.

### 2.2 New file: `server.py` (FastAPI + uvicorn)

**Deps:** `fastapi`, `uvicorn[standard]`, `python-multipart`, `silero-vad`; system **ffmpeg**.

#### Endpoints

| Method | Path | Behavior |
|--------|------|----------|
| `POST` | `/v1/transcriptions` | Multipart audio upload → save under `server_jobs/{job_id}/` → enqueue → return `{job_id}` immediately |
| `GET` | `/v1/transcriptions/{job_id}` | `{status, chunks_done, chunks_total, text, segments?, error?}` — `text` may grow via `progress_cb` |
| `POST` | `/v1/transcriptions/sync` | Short clips only (enforce ≤ 60 s) for curl testing |
| `GET` | `/health` | Model name/path, device, tokenizer path — LAN connectivity check |

#### Server behavior details

- `ThreadPoolExecutor(max_workers=1)` — **serialize GPU**; queue other jobs.  
- Job state = in-memory dict + lock; job dirs on disk; TTL cleanup ~24 h. **No DB in v1.**  
- **Every upload** through ffmpeg first:

  ```bash
  ffmpeg -y -i input -ac 1 -ar 16000 -f wav output.wav
  ```

  Expo records AAC/m4a; soundfile often cannot read it. One decode path for all formats.  
- CORS `allow_origins=["*"]` for early testing.  
- Bind `--host 0.0.0.0 --port 8000`; open Windows Firewall private network; phone on same Wi-Fi (if corporate client isolation, use phone hotspot).  
- Stub `X-API-Key` header check as no-op now; real auth + HTTPS when moving to cloud.

### Phase B gates

| # | Gate | Pass criteria |
|---|------|----------------|
| B1 | Chunking | 2–3 min concatenated Libri wav → ordered timestamped segments |
| B2 | VAD failure path | Force noisy/no-VAD audio → fixed+overlap still returns text |
| B3 | Sync API | `curl` short wav → transcript |
| B4 | Job API | Poll shows `chunks_done` advancing + partial `text` |
| B5 | LAN | Phone browser opens `http://<laptop-ip>:8000/health` |

---

## Lifetime 3 / Phase C — Expo mobile app (`mobile/`)

**Goal:** Record → upload → poll → edit → share. No accounts. Transcripts live on device.

### Stack

- React Native + **Expo SDK ≥ 53** (`npx create-expo-app`)  
- **expo-audio** for recording (`expo-av` audio deprecated)  
- **expo-file-system** (+ `/legacy` upload progress on SDK 54+ if needed)  
- **AsyncStorage** for drafts  
- **expo-sharing** + RN `Share`  
- **expo-keep-awake** while recording  
- **expo-router** navigation  

### Recording defaults

- AAC / m4a ~64 kbps mono → ~30 MB / hour  
- Do **not** force 16 kHz on device; server resamples via ffmpeg  

### Screens

1. **Record** — big button, timer, keep-awake, list of drafts  
2. **Processing** — upload progress → poll job every ~2 s; show chunks progress + growing partial transcript; **persist `jobId` immediately** so kill/reopen resumes polling  
3. **Transcript** — multiline `TextInput`, debounce autosave, Share text / Share `.txt` / Copy  
4. **Settings** — server base URL (default `http://<laptop-ip>:8000`)  

### Draft schema (AsyncStorage)

```json
{
  "jobId": "...",
  "createdAt": "...",
  "title": "...",
  "editedText": "...",
  "status": "recording|uploading|processing|ready|error"
}
```

This is the **entire** v1 storage layer. No backend user DB.

### Platform gotchas

- v1: app should stay **foregrounded** while recording (true background recording = v2: iOS `UIBackgroundModes: audio`, Android foreground service).  
- Plain `http://` works in Expo Go; production builds need cleartext/ATS exceptions **or** HTTPS server.  
- Accept iOS local-network permission prompt.  
- On phone-call interruption: save partial recording if possible.  

### Phase C gates

| # | Gate | Pass criteria |
|---|------|----------------|
| C1 | Happy path | 10 s record → upload → poll → edit → share to email |
| C2 | Long test | 30–60 min podcast-out-loud test end-to-end |
| C3 | Resume | Kill app mid-poll → reopen → resume → finish transcript |

---

## Lifetime 4 — Financial domain adaptation

**Goal:** Better recognition of earnings/finance speech (EBITDA, GAAP, tickers, etc.).

### Data

| Role | Dataset | Notes |
|------|---------|--------|
| General pretrain | LibriSpeech (`train.100` → optionally more) | Lifetime 0 |
| Finance fine-tune | [kensho/SPGISpeech2.0](https://huggingface.co/datasets/kensho/SPGISpeech2.0) | Check **non-commercial** license |
| Finance eval (do not train) | Earnings-21 / Earnings-22 via HF longform leaderboard sets | Report WER |

### Tokenizer upgrade (do this before or at start of fine-tune)

**Recommended:** **BPE** trained on LibriSpeech **+** SPGISpeech transcripts (or SPGISpeech alone if continuing from a BPE Libri run).  

**Alternative:** character tokenizer (can always spell; longer sequences).  

**Not recommended for finance:** keep Libri-only WordLevel.

### Fine-tune procedure (high level)

1. Build/load finance-capable tokenizer; **rebuild model projection** if vocab size changed (cannot naively load old decoder head onto new vocab — either resize carefully or train decoder head from scratch while optionally freezing encoder).  
2. Load LibriSpeech encoder weights when compatible; if vocab changed, map what you can or fine-tune full model with lower LR.  
3. Train on SPGISpeech with lower LR than from-scratch.  
4. Save `checkpoint` + `tokenizer` + `config` as one **bundle** directory.  
5. Point `inference.py` / server `--bundle` at that directory.

### Gate 4

- [ ] Earnings-21 or Earnings-22 WER measured and logged  
- [ ] Spot-check phrases like EBITDA / revenue / guidance in held-out finance audio  
- [ ] Mobile/server still work unchanged except bundle path  

---

## Lifetime 5 — Model iteration (still same product shell)

After finance baseline:

- SpecAugment / better LR schedule  
- Slightly longer windows (e.g. 30 s) **only if VRAM allows** — still chunk long meetings  
- Beam search (optional)  
- Optional hybrid CTC+attention (advanced; not required)  
- Do **not** rewrite the Expo app for each model bump  

---

## Lifetime 6 — Product v2+ (explicitly deferred)

Do **not** build these in v1:

- User accounts / cloud DB for transcripts  
- On-device inference (ONNX / ExecuTorch)  
- Live streaming partial ASR while speaking  
- Speaker diarization  
- Punctuation / truecasing restoration model  
- Beam search / batched decode / KV cache (performance)  
- Android/iOS true background recording  
- HTTPS + real auth + cloud GPU  
- Fancy overlap-merge alignment  

When cloud moves happen: reuse same `inference.py` + job API; put TLS and API keys in front.

---

## 4. Build order (master gate table)

| Order | Work | Gate |
|------:|------|------|
| 0 | Train LibriSpeech on NVIDIA | Checkpoint + tokenizer on disk; non-crash val |
| 1 | `inference.py` + file mode | Libri clip transcribes; mel matches dataset |
| 2 | Mic mode | Read-aloud sentence OK |
| 3 | Chunking (VAD primary + fixed/overlap fallback) | 2–3 min wav ordered segments; noisy fallback works |
| 4 | FastAPI sync | curl short wav → text |
| 5 | FastAPI jobs | poll progress + partial text |
| 6 | LAN health | phone browser hits `/health` |
| 7 | Expo happy path | record → edit → share |
| 8 | Long-recording hardening | 30–60 min + kill/resume |
| 9 | SPGISpeech fine-tune (extend BPE on finance text) | finance WER on Earnings-21/22 |
| 10 | Bundle swap on server | app unchanged; better transcripts |

---

## 5. Dependencies checklist

### Python (add as phases need them)

```
# already roughly present
torch torchaudio datasets tokenizers torchmetrics tqdm tensorboard

# Phase A+
soundfile sounddevice

# Phase B+
fastapi uvicorn[standard] python-multipart
# silero-vad per its current install docs
```

### System

- **ffmpeg** (Windows: `winget install Gyan.FFmpeg`) — required for m4a/AAC uploads  
- NVIDIA drivers + CUDA-matched PyTorch on training/inference machine  

### Mobile

- Node/npm, Expo Go for dev, EAS later for store builds (not v1-critical)

---

## 6. Critical files map

| File | Why it matters |
|------|----------------|
| `config.py` | `n_mels`, `max_audio_len=500`, `seq_len`, paths; fix cwd-relative weight paths for inference |
| `dataset.py` | Canonical mel/pad/mask — inference must clone this exactly |
| `model.py` | `build_transformer(n_mels, ...)`, `Linear` encoder input |
| `train.py` | Training only; donate `greedy_decode` to `inference.py` |
| **New** `inference.py` | Single ASR engine |
| **New** `transcribe_mic.py` | Desktop proof |
| **New** `server.py` | Job API for phone |
| **New** `mobile/` | Expo client |
| Delete after A | `translate.py` |

---

## 7. Design decisions log (do not re-litigate without reason)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Where inference runs | GPU server (laptop first) | Phone too weak for this PyTorch model v1 |
| App framework | React Native + Expo | iOS+Android one codebase |
| v1 storage | On-device only | Prove product before backend |
| Long audio | Chunk ≤15–16 s | Model limit; attention cost |
| Chunk strategy | VAD pack **+** fixed/overlap fallback | Clean cuts when possible; robust when not |
| Tokenizer now | **BPE (~4k)** | Retrain path; `tokenizer_asr_bpe.json` |
| Tokenizer for finance | Same BPE, rebuild on Libri+SPGI text | Must cover EBITDA etc. |
| Train order | Libri → then SPGISpeech | General speech first |
| Sync vs jobs | Jobs for long form | 1 h ≈ minutes of GPU time |

---

## 8. Agent instructions (for Claude / Cursor)

When implementing from this plan:

1. Implement **one gate** at a time; stop and verify before the next.  
2. Prefer extending `inference.py` over duplicating mel/decode logic.  
3. Never import `train.py` from server/inference.  
4. Never “fix” long audio by setting `max_audio_len` to tens of thousands.  
5. Keep mobile ignorant of PyTorch — HTTP only.  
6. When adding finance: ship **tokenizer+checkpoint bundle**; update server load path.  
7. Update this `PLAN.md` only when decisions change; don’t fork shadow plans.  

---

## 9. Success definition

**v1 success:** A user on Wi-Fi records a 30–60 minute session on their phone, gets a transcript back from the GPU laptop, edits typos, and emails a `.txt` — even if WER is mediocre.

**v1.5 success:** Same flow with a SPGISpeech-finetuned bundle and measurably better finance terms (EBITDA, etc.) on Earnings-21/22.

**v2 success:** Cloud-hosted API, accounts optional, faster decode, optional diarization/punctuation.

---

*End of plan. Prefer this document over chat memory when building.*
