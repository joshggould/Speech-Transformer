from model import build_transformer
from config import get_config, get_weights_file_path, latest_weights_file_path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR
import warnings
from tqdm import tqdm
import os
from pathlib import Path
# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
# --- STUDY: original WordLevel tokenizer (replaced by BPE above) ---
# from tokenizers.models import WordLevel
# from tokenizers.trainers import WordLevelTrainer

import torchmetrics
from torch.utils.tensorboard import SummaryWriter
from dataset import AudioDataset, causal_mask
from inference import decode_sequence  # greedy or beam; single copy in inference.py


def run_validation(model, validation_ds, tokenizer, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0
    cfg = get_config()
    beam_size = int(cfg.get("beam_size", 1))
    length_penalty = float(cfg.get("length_penalty", 0.6))

    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)  # (B, T, n_mels)
            encoder_mask = batch["encoder_mask"].to(device)  # (B, 1, 1, T)

            # check that the batch size is 1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = decode_sequence(
                model, encoder_input, encoder_mask, tokenizer, max_len, device,
                beam_size=beam_size, length_penalty=length_penalty,
            )

            target_text = batch["text"][0]
            model_out_text = tokenizer.decode(
                model_out.detach().cpu().tolist(),
                skip_special_tokens=True,
            )

            expected.append(target_text)
            predicted.append(model_out_text)

            print_msg('-' * console_width)
            print_msg(f"{'TARGET: ':>12}{target_text}")
            print_msg(f"{'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-' * console_width)
                break

    if writer:
        # ASR metrics (lower is better)
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

def get_all_sentences(ds):
    # Iterate the text column only -- iterating full rows would decode every
    # audio file (datasets>=4 requires torchcodec for that and it's ~28k
    # decodes wasted just to read transcripts for the tokenizer).
    for text in ds["text"]:
        yield text

def get_or_build_tokenizer(config, ds):
    """Train or load a BPE tokenizer on transcripts (not WordLevel).

    Uses a new filename (tokenizer_asr_bpe.json) so an old WordLevel
    tokenizer_asr.json is never reused by accident. Delete tokenizer_asr_bpe.json
    if you want to rebuild with a different bpe_vocab_size.
    """
    name = config.get("tokenizer_name", "asr_bpe")
    tokenizer_path = Path(config["tokenizer_file"].format(name))
    if not tokenizer_path.exists():
        vocab_size = int(config.get("bpe_vocab_size", 4000))
        print(f"Building BPE tokenizer (vocab_size={vocab_size}) -> {tokenizer_path}")
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            vocab_size=vocab_size,
            min_frequency=2,
            show_progress=True,
        )
        tokenizer.train_from_iterator(get_all_sentences(ds), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
        print(f"Saved BPE tokenizer with {tokenizer.get_vocab_size()} tokens")
    else:
        print(f"Loading existing tokenizer from {tokenizer_path}")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


# --- STUDY: original WordLevel tokenizer builder (kept for comparison) -------
# Whole-word vocab: unseen words (e.g. finance jargon) become [UNK] and cannot
# be spelled from pieces. BPE above fixes that. To revive WordLevel for study:
#   1) uncomment the WordLevel imports at the top of this file
#   2) point config tokenizer_name at "asr" (tokenizer_asr.json)
#   3) replace get_or_build_tokenizer body with the block below
#
# def get_or_build_tokenizer_wordlevel(config, ds):
#     tokenizer_path = Path(config["tokenizer_file"].format("asr"))
#     if not tokenizer_path.exists():
#         tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
#         tokenizer.pre_tokenizer = Whitespace()
#         trainer = WordLevelTrainer(
#             special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
#             min_frequency=2,
#         )
#         tokenizer.train_from_iterator(get_all_sentences(ds), trainer=trainer)
#         tokenizer.save(str(tokenizer_path))
#     else:
#         tokenizer = Tokenizer.from_file(str(tokenizer_path))
#     return tokenizer
# ------------------------------------------------------------------------------

def get_ds(config):
    # It only has the train split, so we divide it overselves
    ds_raw = load_dataset("openslr/librispeech_asr", "clean", split="train.100")

    # Build tokenizers
    tokenizer = get_or_build_tokenizer(config, ds_raw)

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = AudioDataset(train_ds_raw, config['seq_len'], tokenizer, config['max_audio_len'])
    val_ds = AudioDataset(val_ds_raw, config['seq_len'], tokenizer, config['max_audio_len'])

    

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer

def get_model(config, vocab_tgt_len):
    model = build_transformer(
        config["n_mels"],
        vocab_tgt_len,
        config["max_audio_len"],  # encoder positional encoding length
        config["seq_len"],        # decoder positional encoding length
        d_model=config["d_model"],
    )
    return model

def train_model(config):
    # Define the device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: cuda")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: mps")
    else:
        device = torch.device("cpu")
        print("Using device: cpu")
        print("WARNING: CUDA not available. Install a CUDA build of PyTorch to use your NVIDIA GPU.")
        print("  pip uninstall torch -y")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu124")

    # Make sure the weights folder exists
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer = get_ds(config)
    model = get_model(config, tokenizer.get_vocab_size()).to(device)
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        if device.type == "cuda":
            torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, tokenizer.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Run validation at the end of every epoch
        run_validation(model, val_dataloader, tokenizer, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)