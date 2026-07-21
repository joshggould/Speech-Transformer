"""Single audio -> transcript engine (PLAN.md Phase A/B core).

transcribe_mic.py (CLI) and server.py (FastAPI) are thin wrappers around this
module. Import rule from PLAN.md: this file may import model.py, dataset.py,
config.py, tokenizers -- NEVER train.py.
"""
from pathlib import Path

import torch
import torchaudio.functional as AF
import torchaudio.transforms as T
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from config import get_config
from dataset import causal_mask
from model import build_transformer

# All artifact paths resolve against the repo, not the cwd (PLAN.md gotcha:
# config helpers use Path('.') and break when launched from elsewhere).
REPO_DIR = Path(__file__).resolve().parent

SAMPLE_RATE = 16000
HOP_LENGTH = 512


# ---------------------------------------------------------------------------
# Mel features -- must stay identical to AudioDataset (dataset.py)
# ---------------------------------------------------------------------------

def get_mel_transforms():
    mel_transform = T.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        hop_length=512,
        n_mels=80,
        mel_scale="slaney",
    )
    db_transform = T.AmplitudeToDB()
    return mel_transform, db_transform


def prepare_encoder_input(waveform, max_audio_len=500, mel_transform=None, db_transform=None):
    """Waveform (16 kHz mono, 1-D or (1, samples)) -> (encoder_input, encoder_mask).

    Replicates AudioDataset.__getitem__ exactly: mel -> dB -> (time, n_mels)
    -> zero-pad/truncate to max_audio_len, mask marks real frames.
    Verified against AudioDataset by test_mel_parity.py.
    """
    if mel_transform is None or db_transform is None:
        mel_transform, db_transform = get_mel_transforms()

    waveform = torch.as_tensor(waveform, dtype=torch.float32)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # (1, samples) for MelSpectrogram

    mel_spec = mel_transform(waveform)
    mel_spec_db = db_transform(mel_spec)

    # (n_mels, time) -> (time, n_mels)
    encoder_input = mel_spec_db.squeeze(0).transpose(0, 1)

    n_frames = encoder_input.size(0)
    if n_frames > max_audio_len:
        encoder_input = encoder_input[:max_audio_len]
        n_frames = max_audio_len
    elif n_frames < max_audio_len:
        padding = torch.zeros(max_audio_len - n_frames, encoder_input.size(1))
        encoder_input = torch.cat([encoder_input, padding], dim=0)

    encoder_mask = torch.zeros(1, 1, max_audio_len)
    encoder_mask[:, :, :n_frames] = 1
    encoder_mask = encoder_mask.int()

    return encoder_input, encoder_mask


def ensure_16k_mono(waveform, sample_rate):
    """Any waveform (1-D (samples,) or 2-D (channels, samples)) -> 1-D 16 kHz mono float32."""
    waveform = torch.as_tensor(waveform, dtype=torch.float32)
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=0)
    elif waveform.dim() != 1:
        raise ValueError(f"Expected 1-D or 2-D waveform, got shape {tuple(waveform.shape)}")
    if sample_rate != SAMPLE_RATE:
        waveform = AF.resample(waveform, orig_freq=sample_rate, new_freq=SAMPLE_RATE)
    return waveform


def load_audio(path):
    """Audio file (wav/flac/ogg...) -> (waveform 1-D float32 tensor, sample_rate).

    Uses soundfile (stable on Windows; torchaudio I/O backends are in flux).
    m4a/AAC is NOT supported here -- the server converts those via ffmpeg first.
    """
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    waveform = torch.from_numpy(data)
    if waveform.dim() == 2:  # soundfile gives (frames, channels)
        waveform = waveform.transpose(0, 1)
    return waveform, sr


# ---------------------------------------------------------------------------
# Model / tokenizer loading
# ---------------------------------------------------------------------------

_DUMMY_SENTENCES = [
    "THE CAT SAT ON THE MAT",
    "HELLO WORLD THIS IS A TEST",
    "SPEECH TO TEXT PIPELINE CHECK",
]


def build_dummy_tokenizer():
    """Tiny WordLevel tokenizer with the same specials as training -- lets the
    pipeline run before tokenizer_asr.json exists. Output is meaningless."""
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(
        special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=1
    )
    tokenizer.train_from_iterator(_DUMMY_SENTENCES, trainer=trainer)
    return tokenizer


def resolve_tokenizer_path(config):
    return REPO_DIR / config["tokenizer_file"].format("asr")


def resolve_latest_checkpoint(config):
    folder = REPO_DIR / f"{config['datasource']}_{config['model_folder']}"
    files = sorted(folder.glob(f"{config['model_basename']}*"))
    return files[-1] if files else None


def load_model_and_tokenizer(config=None, device=None, checkpoint_path=None,
                             tokenizer_path=None, allow_random_init=True):
    """Returns (model, tokenizer, info).

    Checkpoint + tokenizer must always match (vocab size sets the projection
    layer) -- pass both explicitly when not using the repo defaults.
    With allow_random_init=True the pipeline runs before training has produced
    artifacts: missing tokenizer -> dummy tokenizer, missing checkpoint ->
    random weights. Output is garbage in that mode, by design (pipeline test).
    """
    config = config or get_config()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    tok_path = Path(tokenizer_path) if tokenizer_path else resolve_tokenizer_path(config)
    dummy_tokenizer = not tok_path.exists()
    if not dummy_tokenizer:
        tokenizer = Tokenizer.from_file(str(tok_path))
        tokenizer_source = str(tok_path)
    elif allow_random_init:
        print(f"WARNING: tokenizer not found at {tok_path} -- using a dummy tokenizer. "
              "Transcripts will be meaningless until training produces tokenizer_asr.json.")
        tokenizer = build_dummy_tokenizer()
        tokenizer_source = "dummy (no tokenizer file)"
    else:
        raise FileNotFoundError(f"Tokenizer not found: {tok_path}")

    ckpt_path = Path(checkpoint_path) if checkpoint_path else resolve_latest_checkpoint(config)
    ckpt_exists = ckpt_path is not None and Path(ckpt_path).exists()

    if ckpt_exists and dummy_tokenizer:
        raise RuntimeError(
            f"Found checkpoint {ckpt_path} but no tokenizer at {tok_path}. "
            "A checkpoint can only be decoded with its matching tokenizer -- "
            "pass tokenizer_path explicitly."
        )

    model = build_transformer(
        config["n_mels"],
        tokenizer.get_vocab_size(),
        config["max_audio_len"],
        config["seq_len"],
        d_model=config["d_model"],
    ).to(device)

    if ckpt_exists:
        state = torch.load(str(ckpt_path), map_location=device, weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        checkpoint_source = str(ckpt_path)
    elif allow_random_init:
        print("WARNING: no checkpoint found -- model has RANDOM weights. "
              "Transcripts will be garbage until training saves a .pt file.")
        checkpoint_source = "random-init (no checkpoint)"
    else:
        raise FileNotFoundError(
            f"No checkpoint found (looked for {ckpt_path or 'librispeech_weights/smodel_*'})"
        )

    model.eval()
    info = {
        "device": str(device),
        "tokenizer": tokenizer_source,
        "checkpoint": checkpoint_source,
        "random_weights": not ckpt_exists,
        "dummy_tokenizer": dummy_tokenizer,
        "vocab_size": tokenizer.get_vocab_size(),
    }
    return model, tokenizer, info


# ---------------------------------------------------------------------------
# Greedy decoding (moved here from train.py -- train.py imports it back)
# ---------------------------------------------------------------------------

def greedy_decode(model, source, source_mask, tokenizer, max_len, device):
    sos_idx = tokenizer.token_to_id('[SOS]')
    eos_idx = tokenizer.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token (must be int token ids, not mel dtype)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type(torch.long).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(decoder_input).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


# ---------------------------------------------------------------------------
# Long-audio chunking (PLAN.md 2.1: VAD packing primary, fixed+overlap fallback)
# ---------------------------------------------------------------------------

_SILERO_MODEL = None


def _silero_segments(waveform):
    """Speech segments [(start_sample, end_sample)] via silero-vad."""
    global _SILERO_MODEL
    from silero_vad import get_speech_timestamps, load_silero_vad

    if _SILERO_MODEL is None:
        _SILERO_MODEL = load_silero_vad()
    stamps = get_speech_timestamps(waveform, _SILERO_MODEL, sampling_rate=SAMPLE_RATE)
    return [(s["start"], s["end"]) for s in stamps]


def _energy_segments(waveform, frame_ms=30, threshold_db=30.0):
    """Fallback VAD: frames whose RMS is within threshold_db of the loudest frame."""
    frame = int(SAMPLE_RATE * frame_ms / 1000)
    n_frames = len(waveform) // frame
    if n_frames == 0:
        return []
    trimmed = waveform[: n_frames * frame].reshape(n_frames, frame)
    rms_db = 20 * torch.log10(trimmed.pow(2).mean(dim=1).sqrt() + 1e-10)
    active = rms_db > (rms_db.max() - threshold_db)
    segments = []
    start = None
    for i, on in enumerate(active.tolist()):
        if on and start is None:
            start = i * frame
        elif not on and start is not None:
            segments.append((start, i * frame))
            start = None
    if start is not None:
        segments.append((start, len(waveform)))
    return segments


def _merge_segments(segments, max_gap_s=0.3):
    if not segments:
        return []
    max_gap = int(max_gap_s * SAMPLE_RATE)
    merged = [list(segments[0])]
    for s, e in segments[1:]:
        if s - merged[-1][1] < max_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [tuple(m) for m in merged]


def _split_long_segment(waveform, start, end, max_samples):
    """A continuous speech region longer than one chunk: hard-split at the
    lowest-energy frame in the 80-100% region of the window (PLAN.md 2.1.7)."""
    pieces = []
    frame = int(SAMPLE_RATE * 0.03)
    while end - start > max_samples:
        lo = start + int(0.8 * max_samples)
        hi = start + max_samples
        window = waveform[lo:hi]
        n = len(window) // frame
        if n > 0:
            rms = window[: n * frame].reshape(n, frame).pow(2).mean(dim=1)
            cut = lo + int(rms.argmin().item()) * frame
        else:
            cut = hi
        pieces.append((start, cut))
        start = cut
    pieces.append((start, end))
    return pieces


def _pack_segments(segments, max_samples):
    """Greedily pack speech segments into chunks <= max_samples, cutting only at silence."""
    chunks = []
    cur_start, cur_end = None, None
    for s, e in segments:
        if cur_start is None:
            cur_start, cur_end = s, e
        elif e - cur_start <= max_samples:
            cur_end = e
        else:
            chunks.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    if cur_start is not None:
        chunks.append((cur_start, cur_end))
    return chunks


def plan_chunks(waveform, vad_mode="silero", chunk_seconds=15.0, chunk_overlap_seconds=1.5):
    """Waveform (1-D, 16 kHz) -> (chunks [(start_sample, end_sample)], mode_used).

    silero/energy: VAD segments merged (<0.3 s gaps), long regions split at
    low-energy points, packed into <=chunk_seconds chunks at silence boundaries
    (no overlap needed). If VAD finds nothing -> fixed windows with overlap
    (required safety net per PLAN.md 2.1) -- never returns an empty plan.
    """
    # Never exceed what the encoder can see: max_audio_len frames.
    hard_cap = (get_config()["max_audio_len"] - 1) * HOP_LENGTH
    max_samples = min(int(chunk_seconds * SAMPLE_RATE), hard_cap)
    n = len(waveform)
    if n <= max_samples:
        return [(0, n)], "single"

    segments = []
    mode = vad_mode
    if vad_mode == "silero":
        try:
            segments = _silero_segments(waveform)
        except Exception as exc:  # unavailable/failed -> energy fallback
            print(f"WARNING: silero-vad failed ({exc}); falling back to energy VAD")
            mode = "energy"
    if mode == "energy":
        segments = _energy_segments(waveform)

    if mode != "fixed" and segments:
        segments = _merge_segments(segments)
        split = []
        for s, e in segments:
            split.extend(_split_long_segment(waveform, s, e, max_samples))
        return _pack_segments(split, max_samples), mode

    # Fixed windows + overlap: VAD found nothing (noise/music/crosstalk) or forced.
    stride = max_samples - int(chunk_overlap_seconds * SAMPLE_RATE)
    stride = max(stride, int(1.0 * SAMPLE_RATE))
    chunks = []
    start = 0
    while start < n:
        chunks.append((start, min(start + max_samples, n)))
        if start + max_samples >= n:
            break
        start += stride
    return chunks, "fixed"


def _stitch(segments, dedup=False, max_dedup_words=6):
    """Join chunk texts in order. In fixed+overlap mode, naively drop leading
    words of a chunk that repeat the previous chunk's trailing words."""
    texts = []
    for seg in segments:
        words = seg["text"].split()
        if dedup and texts and words:
            prev = texts[-1].split()
            for k in range(min(max_dedup_words, len(prev), len(words)), 0, -1):
                if prev[-k:] == words[:k]:
                    words = words[k:]
                    break
        if words:
            texts.append(" ".join(words))
    return " ".join(texts)


# ---------------------------------------------------------------------------
# Top-level transcription
# ---------------------------------------------------------------------------

def transcribe_waveform(model, tokenizer, waveform, sample_rate, device=None,
                        vad_mode="silero", chunk_seconds=15.0,
                        chunk_overlap_seconds=1.5, progress_cb=None):
    """Any-length waveform -> {"text", "segments": [{start, end, text}], "chunk_mode"}.

    progress_cb(chunks_done, chunks_total, partial_text) fires after each chunk
    (used by the server to expose a growing transcript).
    """
    config = get_config()
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)

    wave = ensure_16k_mono(waveform, sample_rate)
    chunks, mode = plan_chunks(wave, vad_mode=vad_mode, chunk_seconds=chunk_seconds,
                               chunk_overlap_seconds=chunk_overlap_seconds)
    dedup = mode == "fixed"

    mel_transform, db_transform = get_mel_transforms()
    segments = []
    with torch.inference_mode():
        for i, (s, e) in enumerate(chunks):
            encoder_input, encoder_mask = prepare_encoder_input(
                wave[s:e], config["max_audio_len"], mel_transform, db_transform
            )
            source = encoder_input.unsqueeze(0).to(device)       # (1, T, n_mels)
            source_mask = encoder_mask.unsqueeze(0).to(device)   # (1, 1, 1, T)
            ids = greedy_decode(model, source, source_mask, tokenizer,
                                config["seq_len"], device)
            text = tokenizer.decode([int(t) for t in ids.detach().cpu().tolist()])
            segments.append({"start": s / SAMPLE_RATE, "end": e / SAMPLE_RATE, "text": text})
            if progress_cb:
                progress_cb(i + 1, len(chunks), _stitch(segments, dedup=dedup))

    return {"text": _stitch(segments, dedup=dedup), "segments": segments, "chunk_mode": mode}


def transcribe_file(path, model, tokenizer, **kwargs):
    waveform, sr = load_audio(path)
    return transcribe_waveform(model, tokenizer, waveform, sr, **kwargs)
