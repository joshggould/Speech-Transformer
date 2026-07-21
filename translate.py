"""DEPRECATED (PLAN.md Phase A): stale text-translation leftover.

This script referenced lang_src/lang_tgt config keys that no longer exist and
never worked for speech. Use instead:

    python transcribe_mic.py --file path\\to\\clip.wav
    python transcribe_mic.py --mic --seconds 10

Quarantined rather than deleted until Gate A1 passes with real trained
weights; safe to delete after that.
"""
raise SystemExit(__doc__)
