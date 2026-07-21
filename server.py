"""FastAPI transcription server (PLAN.md Phase B / Lifetime 2).

Job-based API: long audio takes minutes of GPU time, so uploads return a
job_id immediately and clients poll for progress + growing partial text.

Run:  uvicorn server:app --host 0.0.0.0 --port 8000
  or: python server.py

Endpoints:
    GET  /health                     model/device info, LAN connectivity check
    POST /v1/transcriptions          multipart upload -> {"job_id"}
    GET  /v1/transcriptions/{id}     {status, chunks_done, chunks_total, text, ...}
    POST /v1/transcriptions/sync     short clips (<=60s) inline, for curl testing

Phone uploads are AAC/m4a -> converted via ffmpeg (winget install Gyan.FFmpeg).
Without ffmpeg only wav/flac/ogg uploads work (soundfile).
"""
import shutil
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from inference import (
    REPO_DIR,
    SAMPLE_RATE,
    ensure_16k_mono,
    load_audio,
    load_model_and_tokenizer,
    transcribe_waveform,
)

JOBS_DIR = REPO_DIR / "server_jobs"
JOB_TTL_SECONDS = 24 * 3600
MAX_SYNC_SECONDS = 60.0
SOUNDFILE_OK_SUFFIXES = {".wav", ".flac", ".ogg"}

JOBS = {}
JOBS_LOCK = threading.Lock()
# One worker: the GPU must be serialized; queued jobs wait their turn.
EXECUTOR = ThreadPoolExecutor(max_workers=1)

MODEL = None
TOKENIZER = None
MODEL_INFO = {}


def require_api_key(x_api_key: str | None = Header(default=None)):
    """No-op auth stub (PLAN.md): becomes a real key check when leaving the LAN."""
    return True


def _cleanup_old_jobs():
    if not JOBS_DIR.exists():
        return
    cutoff = time.time() - JOB_TTL_SECONDS
    for job_dir in JOBS_DIR.iterdir():
        if job_dir.is_dir() and job_dir.stat().st_mtime < cutoff:
            shutil.rmtree(job_dir, ignore_errors=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, TOKENIZER, MODEL_INFO
    MODEL, TOKENIZER, MODEL_INFO = load_model_and_tokenizer()
    if MODEL_INFO["random_weights"]:
        print("=" * 70)
        print("SERVER RUNNING WITH RANDOM WEIGHTS -- transcripts will be garbage")
        print("until training produces a checkpoint + tokenizer_asr.json.")
        print("=" * 70)
    JOBS_DIR.mkdir(exist_ok=True)
    _cleanup_old_jobs()
    yield
    EXECUTOR.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="Speech-Transformer ASR", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


def _decode_upload(path: Path):
    """Uploaded audio file -> 1-D 16 kHz mono waveform tensor.

    Everything goes through ffmpeg when available (one decode path for wav /
    flac / m4a / anything). Without ffmpeg, soundfile handles wav/flac/ogg only.
    """
    if shutil.which("ffmpeg"):
        wav_path = path.with_name("converted_16k.wav")
        proc = subprocess.run(
            ["ffmpeg", "-y", "-i", str(path), "-ac", "1", "-ar", str(SAMPLE_RATE),
             "-f", "wav", str(wav_path)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise ValueError(f"ffmpeg could not decode the audio: {proc.stderr[-500:]}")
        waveform, sr = load_audio(wav_path)
    else:
        if path.suffix.lower() not in SOUNDFILE_OK_SUFFIXES:
            raise ValueError(
                f"'{path.suffix}' uploads need ffmpeg on the server "
                "(winget install Gyan.FFmpeg). Without it only wav/flac/ogg work."
            )
        waveform, sr = load_audio(path)
    return ensure_16k_mono(waveform, sr)


def _save_upload(upload: UploadFile, job_dir: Path) -> Path:
    suffix = Path(upload.filename or "audio").suffix or ".bin"
    dest = job_dir / f"upload{suffix}"
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dest


def _run_job(job_id: str, audio_path: Path):
    def progress(done, total, partial_text):
        with JOBS_LOCK:
            job = JOBS[job_id]
            job["chunks_done"] = done
            job["chunks_total"] = total
            job["text"] = partial_text

    try:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "processing"
        waveform = _decode_upload(audio_path)
        result = transcribe_waveform(MODEL, TOKENIZER, waveform, SAMPLE_RATE,
                                     progress_cb=progress)
        with JOBS_LOCK:
            JOBS[job_id].update(
                status="done", text=result["text"], segments=result["segments"],
                chunk_mode=result["chunk_mode"],
                chunks_done=len(result["segments"]), chunks_total=len(result["segments"]),
            )
    except Exception as exc:
        with JOBS_LOCK:
            JOBS[job_id].update(status="error", error=str(exc))


@app.get("/health")
def health():
    with JOBS_LOCK:
        active = sum(1 for j in JOBS.values() if j["status"] in ("queued", "processing"))
    return {
        "status": "ok",
        **MODEL_INFO,
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "active_jobs": active,
    }


@app.post("/v1/transcriptions")
def create_transcription(file: UploadFile = File(...), _=Depends(require_api_key)):
    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)
    audio_path = _save_upload(file, job_dir)
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "queued", "chunks_done": 0, "chunks_total": None,
            "text": "", "segments": None, "error": None, "created": time.time(),
        }
    EXECUTOR.submit(_run_job, job_id, audio_path)
    return {"job_id": job_id}


@app.get("/v1/transcriptions/{job_id}")
def get_transcription(job_id: str, _=Depends(require_api_key)):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job_id")
        return dict(job)


@app.post("/v1/transcriptions/sync")
def transcribe_sync(file: UploadFile = File(...), _=Depends(require_api_key)):
    job_dir = JOBS_DIR / f"sync_{uuid.uuid4().hex[:8]}"
    job_dir.mkdir(parents=True)
    try:
        audio_path = _save_upload(file, job_dir)
        try:
            waveform = _decode_upload(audio_path)
        except ValueError as exc:
            raise HTTPException(status_code=415, detail=str(exc))
        duration = len(waveform) / SAMPLE_RATE
        if duration > MAX_SYNC_SECONDS:
            raise HTTPException(
                status_code=413,
                detail=f"{duration:.0f}s audio too long for sync (max {MAX_SYNC_SECONDS:.0f}s); "
                       "use POST /v1/transcriptions",
            )
        # Through the same single-worker executor so the GPU stays serialized.
        future = EXECUTOR.submit(transcribe_waveform, MODEL, TOKENIZER,
                                 waveform, SAMPLE_RATE)
        return future.result()
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
