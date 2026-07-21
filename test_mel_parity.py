"""Gate A2 (PLAN.md): inference mel prep must match AudioDataset exactly.

Feeds identical waveforms through AudioDataset.__getitem__ and
inference.prepare_encoder_input and asserts encoder_input/encoder_mask parity.
Covers the pad path (short audio) and the truncate path (long audio).

Run: python test_mel_parity.py
"""
import numpy as np
import torch

from config import get_config
from dataset import AudioDataset
from inference import build_dummy_tokenizer, prepare_encoder_input


class FakeHFDS:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def main():
    config = get_config()
    rng = np.random.default_rng(42)

    cases = {
        "short (pad path, 1.5s)": 1.5,
        "exact-ish (15.9s)": 15.9,
        "long (truncate path, 20s)": 20.0,
    }

    items, names = [], []
    for name, seconds in cases.items():
        array = (rng.standard_normal(int(16000 * seconds)) * 0.05).astype(np.float64)
        items.append({"audio": {"array": array, "sampling_rate": 16000},
                      "text": "THE CAT SAT ON THE MAT"})
        names.append(name)

    dataset = AudioDataset(FakeHFDS(items), seq_len=config["seq_len"],
                           tokenizer=build_dummy_tokenizer(),
                           max_audio_len=config["max_audio_len"])

    failures = 0
    for idx, name in enumerate(names):
        expected = dataset[idx]
        got_input, got_mask = prepare_encoder_input(
            items[idx]["audio"]["array"], config["max_audio_len"]
        )

        input_ok = torch.allclose(expected["encoder_input"], got_input, atol=1e-5)
        mask_ok = torch.equal(expected["encoder_mask"], got_mask)
        max_diff = (expected["encoder_input"] - got_input).abs().max().item()

        status = "OK" if (input_ok and mask_ok) else "FAIL"
        print(f"[{status}] {name}: max_mel_diff={max_diff:.2e} mask_match={mask_ok}")
        if not (input_ok and mask_ok):
            failures += 1

    if failures:
        print(f"MEL PARITY FAILED ({failures}/{len(names)} cases)")
        raise SystemExit(1)
    print("MEL PARITY PASSED")


if __name__ == "__main__":
    main()
