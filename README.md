# Speech Transformer

Mel spectrogram → Transformer encoder → text decoder (ASR), adapted from a from-scratch translation Transformer.

## Setup

```bash
git clone https://github.com/joshggould/Speech-Transformer.git
cd Speech-Transformer
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# On NVIDIA: install CUDA build of torch/torchaudio from pytorch.org
```

## Quick smoke test (no dataset download)

```bash
python smoke_test_offline.py
```

## Train (LibriSpeech clean train.100)

```bash
python train.py
```

## Layout

| File | Role |
|------|------|
| `dataset.py` | `AudioDataset` — mel encoder input + transcript decoder |
| `model.py` | Transformer with `Linear(n_mels → d_model)` on encoder |
| `train.py` | LibriSpeech training loop |
| `config.py` | Hyperparameters |

Translation reference project is kept separate; this repo is speech-only.

## Planned next

1. Train general ASR on LibriSpeech (NVIDIA)
2. Fine-tune on financial speech (e.g. SPGISpeech)
3. Evaluate on Earnings-21 / Earnings-22
