"""Offline smoke test: synthetic audio+text (no Hugging Face download).
Checks AudioDataset mel path + Speech Transformer forward shapes.
"""
import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from config import get_config
from dataset import AudioDataset
from model import build_transformer


class FakeHFDS:
    """Minimal stand-in for a Hugging Face dataset."""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def build_tokenizer(texts):
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(
        special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
        min_frequency=1,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer


def main():
    config = get_config()
    device = torch.device("cpu")
    print(f"device: {device}")

    # ~1–2 seconds of fake 16 kHz audio
    rng = np.random.default_rng(0)
    items = []
    texts = [
        "THE CAT SAT ON THE MAT",
        "HELLO WORLD THIS IS A TEST",
    ]
    for text in texts:
        samples = int(16000 * 1.5)
        array = rng.standard_normal(samples).astype(np.float32) * 0.01
        items.append({
            "audio": {"array": array, "sampling_rate": 16000},
            "text": text,
        })

    ds = FakeHFDS(items)
    tokenizer = build_tokenizer(texts)
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
            print(f"  {k}: {v!r}")

    encoder_input = torch.stack([dataset[0]["encoder_input"], dataset[1]["encoder_input"]])
    encoder_mask = torch.stack([dataset[0]["encoder_mask"], dataset[1]["encoder_mask"]])
    decoder_input = torch.stack([dataset[0]["decoder_input"], dataset[1]["decoder_input"]])
    decoder_mask = torch.stack([dataset[0]["decoder_mask"], dataset[1]["decoder_mask"]])
    label = torch.stack([dataset[0]["label"], dataset[1]["label"]])

    model = build_transformer(
        config["n_mels"],
        tokenizer.get_vocab_size(),
        config["max_audio_len"],
        config["seq_len"],
        d_model=config["d_model"],
        N=2,
        h=4,
        d_ff=512,
    )
    model.eval()

    with torch.no_grad():
        enc = model.encode(encoder_input, encoder_mask)
        dec = model.decode(enc, encoder_mask, decoder_input, decoder_mask)
        logits = model.project(dec)

    print("forward pass shapes:")
    print(f"  encoder_input: {tuple(encoder_input.shape)}")
    print(f"  encoder_out:   {tuple(enc.shape)}")
    print(f"  decoder_out:   {tuple(dec.shape)}")
    print(f"  logits:        {tuple(logits.shape)}")
    print(f"  label:         {tuple(label.shape)}")
    assert encoder_input.shape == (2, config["max_audio_len"], config["n_mels"])
    assert logits.shape == (2, config["seq_len"], tokenizer.get_vocab_size())
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
