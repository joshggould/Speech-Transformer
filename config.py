from pathlib import Path

def get_config():
    return {
        "batch_size": 8,  # audio batches are heavier than text; start smaller
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 350,  # max transcript token length
        "max_audio_len": 500,  # max mel time frames (must match AudioDataset)
        "d_model": 512,
        "n_mels": 80,
        "datasource": "librispeech_bpe",  # separate weights folder from WordLevel run
        "hf_dataset": "openslr/librispeech_asr",
        "model_folder": "weights",
        "model_basename": "smodel_",
        "preload": None,  # must be None for BPE retrain (old WordLevel ckpts won't load)
        "tokenizer_name": "asr_bpe",  # -> tokenizer_asr_bpe.json (not the old WordLevel file)
        "tokenizer_file": "tokenizer_{0}.json",
        "bpe_vocab_size": 4000,  # good ASR starting point; try 2000–5000
        "beam_size": 4,  # 1 = greedy; 4–8 typical for better CER/WER at test time
        "length_penalty": 0.6,  # beam score /= len**penalty (discourage too-short hyps)
        "experiment_name": "runs/speech_transformer_bpe"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])