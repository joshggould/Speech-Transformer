import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchaudio.transforms as T


class AudioDataset(Dataset):
    
    def __init__(self, ds, seq_len, tokenizer, max_audio_len=500):
        super().__init__()
        self.ds = ds
        self.seq_len = seq_len
        self.tokenizer = tokenizer 
        self.max_audio_len = max_audio_len
        self.sos_token = torch.tensor([tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer.token_to_id("[PAD]")], dtype=torch.int64)
        self.mel_transform = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=512,
            n_mels=80,
            mel_scale="slaney",
        )
        self.db_transform = T.AmplitudeToDB()

    def __len__(self):
        return len(self.ds)


    def __getitem__(self, idx):
        #extract the pair of audio and target transcript 
        pair = self.ds[idx]
        #get teh audio seperate 
        audio = pair["audio"]
        #get the target text 
        text = pair["text"]
        #get the actuall waveform values from the audio
        waveform = audio["array"]
        #encode the transcripts to tokens
        dec_input_tokens = self.tokenizer.encode(text).ids
        #how much padding to hit the sentence length 
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1
        #raise error if the length is to long and get negative number
        if dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")
        #concatenate the start of sentence at the beging of teh decoded sentecne plus padding if needed 
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )
        #concatenate the  decoded sentecne with end of sentence plus padding if needed 
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        waveform = torch.tensor(waveform, dtype=torch.float32)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # (1, samples) for MelSpectrogram

        mel_spec = self.mel_transform(waveform)
        mel_spec_db = self.db_transform(mel_spec)

        # (n_mels, time) -> (time, n_mels) for the Transformer encoder
        encoder_input = mel_spec_db.squeeze(0).transpose(0, 1)

        # n_frames = real mel length before pad/truncate (don't name this T — that shadows transforms as T)
        n_frames = encoder_input.size(0)
        if n_frames > self.max_audio_len:
            # Too long: keep only the first max_audio_len frames
            encoder_input = encoder_input[:self.max_audio_len]
            n_frames = self.max_audio_len
        elif n_frames < self.max_audio_len:
            # Too short: pad time with zeros so every example has the same shape
            num_pad = self.max_audio_len - n_frames
            padding = torch.zeros(num_pad, encoder_input.size(1))
            encoder_input = torch.cat([encoder_input, padding], dim=0)

        # Mask marks real frames (1) vs pad frames (0)
        encoder_mask = torch.zeros(1, 1, self.max_audio_len)
        encoder_mask[:, :, :n_frames] = 1
        encoder_mask = encoder_mask.int()

        return {
            "encoder_input": encoder_input,  # (max_audio_len, n_mels)
            "decoder_input": decoder_input,  # (seq_len,)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label": label,  # (seq_len,)
            "text": text,
            "encoder_mask": encoder_mask,  # (1, 1, max_audio_len)
        }

class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Add sos, eos and padding to each sentence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # We will add <s> and </s>
        # We will only add <s>, and </s> only on the label
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Add <s> and </s> token
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only <s> token
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only </s> token
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
    
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0