"""Desktop transcription CLI (PLAN.md Phase A). Replaces the stale translate.py.

Examples:
    python transcribe_mic.py --file clip.flac
    python transcribe_mic.py --file meeting.wav --segments --vad energy
    python transcribe_mic.py --mic --seconds 10
    python transcribe_mic.py --list-devices

Works before training finishes: with no checkpoint/tokenizer on disk it runs a
random-init model + dummy tokenizer (garbage text, but proves the pipeline).
"""
import argparse
import sys

import torch

from inference import (
    SAMPLE_RATE,
    ensure_16k_mono,
    load_audio,
    load_model_and_tokenizer,
    transcribe_waveform,
)


def record_mic(seconds, device_index=None):
    """Record from the microphone -> (waveform 1-D float32 tensor, sample_rate)."""
    import numpy as np
    import sounddevice as sd

    def _record(rate):
        print(f"Recording {seconds:.0f}s at {rate} Hz... speak now")
        data = sd.rec(int(seconds * rate), samplerate=rate, channels=1,
                      dtype="float32", device=device_index)
        sd.wait()
        print("Recording done.")
        return data

    try:
        data, rate = _record(SAMPLE_RATE), SAMPLE_RATE
    except sd.PortAudioError as exc:
        # Some Windows drivers refuse 16 kHz capture -> use device default, resample later.
        info = sd.query_devices(device_index, "input")
        rate = int(info["default_samplerate"])
        print(f"16 kHz capture failed ({exc}); retrying at device default {rate} Hz")
        try:
            data = _record(rate)
        except sd.PortAudioError:
            print("\nMicrophone capture failed. Available devices:\n")
            print(sd.query_devices())
            print("\nPick an input device with --device N")
            raise SystemExit(1)

    waveform = torch.from_numpy(np.ascontiguousarray(data.squeeze()))
    return waveform, rate


def main():
    parser = argparse.ArgumentParser(description="Transcribe an audio file or microphone recording.")
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--file", help="Path to a wav/flac audio file")
    source.add_argument("--mic", action="store_true", help="Record from the microphone")
    source.add_argument("--list-devices", action="store_true", help="List audio input devices and exit")
    parser.add_argument("--seconds", type=float, default=10.0, help="Mic recording length (default 10)")
    parser.add_argument("--device", type=int, default=None, help="Input device index (see --list-devices)")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint .pt path (default: latest in librispeech_weights/)")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer json path (default: tokenizer_asr.json)")
    parser.add_argument("--vad", choices=["silero", "energy", "fixed"], default="silero",
                        help="Chunking mode for audio longer than ~15s (default silero)")
    parser.add_argument("--chunk-seconds", type=float, default=15.0)
    parser.add_argument("--overlap-seconds", type=float, default=1.5, help="Overlap in fixed-window fallback mode")
    parser.add_argument("--segments", action="store_true", help="Print per-chunk timestamps")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    args = parser.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return

    if not args.file and not args.mic:
        parser.error("one of --file / --mic / --list-devices is required")

    device = torch.device("cpu") if args.cpu else None
    model, tokenizer, info = load_model_and_tokenizer(
        device=device, checkpoint_path=args.checkpoint, tokenizer_path=args.tokenizer
    )
    print(f"device:     {info['device']}")
    print(f"checkpoint: {info['checkpoint']}")
    print(f"tokenizer:  {info['tokenizer']}")

    if args.file:
        waveform, rate = load_audio(args.file)
        print(f"file:       {args.file}")
    else:
        waveform, rate = record_mic(args.seconds, args.device)

    duration = len(ensure_16k_mono(waveform, rate)) / SAMPLE_RATE
    print(f"duration:   {duration:.1f}s")

    def progress(done, total, _partial):
        if total > 1:
            print(f"  chunk {done}/{total}", file=sys.stderr)

    result = transcribe_waveform(
        model, tokenizer, waveform, rate,
        vad_mode=args.vad, chunk_seconds=args.chunk_seconds,
        chunk_overlap_seconds=args.overlap_seconds, progress_cb=progress,
    )

    print(f"chunk mode: {result['chunk_mode']} ({len(result['segments'])} chunk(s))")
    if args.segments:
        for seg in result["segments"]:
            print(f"[{seg['start']:8.1f}s - {seg['end']:8.1f}s] {seg['text']}")
    print("TRANSCRIPT:")
    print(result["text"] if result["text"].strip() else "(empty)")


if __name__ == "__main__":
    main()
