"""Smoke test: one batch through AudioDataset + Speech Transformer.
Uses a tiny LibriSpeech slice so we don't download train.100 on the Mac.
"""
from pathlib import Path

import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from config import get_config
from dataset import AudioDataset
from model import build_transformer


def build_tiny_tokenizer(texts, path="tokenizer_asr_smoke.json"):
    if Path(path).exists():
        return Tokenizer.from_file(path)
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(
        special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
        min_frequency=1,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.save(path)
    return tokenizer


def main():
    config = get_config()
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"device: {device}")

    print("Loading tiny LibriSpeech clean validation slice...")
    ds = load_dataset("openslr/librispeech_asr", "clean", split="validation")
    ds = ds.select(range(min(8, len(ds))))
    print(f"examples: {len(ds)}")

    tokenizer = build_tiny_tokenizer([ex["text"] for ex in ds])
    dataset = AudioDataset(
        ds,
        seq_len=config["seq_len"],
        tokenizer=tokenizer,
        max_audio_len=config["max_audio_len"],
    )

    batch = dataset[0]
    print("single example shapes:")
    for k, v in batch.items():
        if hasattr(v, "shape"):
            print(f"  {k}: {tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"  {k}: {v[:80]!r}...")

    # Fake a batch of size 2
    b0, b1 = dataset[0], dataset[1]
    encoder_input = torch.stack([b0["encoder_input"], b1["encoder_input"]]).to(device)
    encoder_mask = torch.stack([b0["encoder_mask"], b1["encoder_mask"]]).to(device)
    decoder_input = torch.stack([b0["decoder_input"], b1["decoder_input"]]).to(device)
    decoder_mask = torch.stack([b0["decoder_mask"], b1["decoder_mask"]]).to(device)
    label = torch.stack([b0["label"], b1["label"]]).to(device)

    model = build_transformer(
        config["n_mels"],
        tokenizer.get_vocab_size(),
        config["max_audio_len"],
        config["seq_len"],
        d_model=config["d_model"],
        N=2,  # smaller for smoke test
        h=4,
        d_ff=512,
    ).to(device)
    model.eval()

    with torch.no_grad():
        enc = model.encode(encoder_input, encoder_mask)
        dec = model.decode(enc, encoder_mask, decoder_input, decoder_mask)
        logits = model.project(dec)

    print("forward pass shapes:")
    print(f"  encoder_input: {tuple(encoder_input.shape)}")
    print(f"  encoder_out:   {tuple(enc.shape)}")
    print(f"  decoder_out:   {tuple(dec.shape)}")
    print(f"  logits:        {tuple(logits.shape)}  (expect B, seq_len, vocab)")
    print(f"  label:         {tuple(label.shape)}")
    assert logits.shape == (2, config["seq_len"], tokenizer.get_vocab_size())
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
