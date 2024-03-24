import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from torch import LongTensor
from torch.utils.data import Dataset

from audio.injectors import TortoiseDiscreteTokenInjector, TortoiseMelSpectrogramInjector


def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split_char) for line in f]
    return filepaths_and_text


def get_prompt_slice(audio, max_audio_length=20, min_audio_length=3, sample_rate=24000, is_eval=False):
    max_sample_length = max_audio_length * sample_rate
    min_sample_length = min_audio_length * sample_rate
    rel_clip = audio
    # if eval uses a middle size sample when it is possible to be more reproducible
    if is_eval:
        sample_length = int((min_sample_length + max_sample_length) / 2)
    else:
        sample_length = random.randint(min_sample_length, max_sample_length)
    gap = rel_clip.shape[-1] - sample_length
    if gap < 0 and is_eval:
        sample_length = rel_clip.shape[-1]
    elif gap < 0:
        sample_length = rel_clip.shape[-1] // 2
    gap = rel_clip.shape[-1] - sample_length

    # if eval start always from the position 0 to be more reproducible
    if is_eval:
        rand_start = 0
    else:
        rand_start = random.randint(0, gap)

    rand_end = rand_start + sample_length
    rel_clip = rel_clip[:, rand_start:rand_end]
    return rel_clip


class GptDataset(Dataset):
    def __init__(self, config, tokenizer, dvae_path, injector, is_eval=False):
        self.is_eval = is_eval
        if not self.is_eval:
            self.path = config['gpt_train']['train_file']
        else:
            self.path = config['gpt_train']['valid_file']
        self.tokenizer = tokenizer
        self.audiopath_and_text = parse_filelist(self.path)
        self.sample_rate = config['vae_train']['sample_rate']
        self.n_mels = config['vae_train']['n_mels']
        self.power = config['vae_train']['power']

        try:
            self.mel_fmax = config['vae_train']['mel_fmax']
        except:
            self.mel_fmax = None

        self.inj = injector
        self.mel_path = config['gpt_train']['mel_dir']
        self.dvae_path = dvae_path
        self.code_inj = TortoiseDiscreteTokenInjector({'in': 'mel', 'out': 'codes'}, self.dvae_path,
                                                      channels=self.n_mels)

    def get_text(self, text):
        tokens = self.tokenizer.encode(text)
        tokens = torch.IntTensor(tokens)
        return tokens

    def __getitem__(self, index):
        audiopath_and_text = self.audiopath_and_text[index]
        wav_file, text = audiopath_and_text[0], audiopath_and_text[1]

        tseq = self.get_text(text)

        # The stop token should always be sacred.
        if torch.any(tseq == 0):
            raise Exception(f"Stop token found in {text}")

        if torch.any(tseq == 1):
            raise Exception(f"[UNK] token found in ===> {text} -> {self.tokenizer.decode(tseq)}")

        audio, sr = torchaudio.load(wav_file)

        if audio.shape[0] > 1:
            audio = audio[0].unsqueeze(0)
        audio = torchaudio.transforms.Resample(sr, self.sample_rate)(audio).cuda()

        """base_name = Path(wav_file).stem

        if not os.path.exists(f'{self.mel_path}/{base_name}.mel.pth'):
            mel = self.inj({'wav': audio.unsqueeze(0)})['mel']
            torch.save(mel.cpu().detach(), f'{self.mel_path}/{base_name}.mel.pth')

        mel = torch.load(f'{self.mel_path}/{base_name}.mel.pth')

        if not os.path.exists(f'{self.mel_path}/{base_name}.melvq.pth'):
            # print(base_name)
            code = self.code_inj({'mel': mel.to('cuda')})['codes']
            torch.save(code, f'{self.mel_path}/{base_name}.melvq.pth')

        qmel = torch.load(f'{self.mel_path}/{base_name}.melvq.pth')"""

        mel = self.inj({'wav': audio.unsqueeze(0)})['mel']
        qmel = self.code_inj({'mel': mel.to('cuda')})['codes']
        mel = mel[0]
        qmel = qmel[0]
        wav_length = mel.shape[1] * 256

        split = random.randint(int(mel.shape[1] // 3), int(mel.shape[1] // 3 * 2))
        if random.random() > 0.5:
            cond_mel = mel[:, :split]
        else:
            cond_mel = mel[:, split:]

        if tseq.shape[0] > 400 or qmel.shape[0] > 600:
            print(f"Warning: {text} text len {tseq.shape[0]} exceed 400 , qmel len {qmel.shape[0]} exceed 600.")
            return None

        return tseq, qmel, cond_mel, wav_length, audio, mel

    def __len__(self):
        return len(self.audiopath_and_text)

    def collate_fn(self, batch):
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            return None

        text_lens = [len(x[0]) for x in batch]
        max_text_len = max(text_lens)

        qmel_lens = [len(x[1]) for x in batch]
        max_qmel_len = max(qmel_lens)

        cond_mel_lens = [x[2].shape[1] for x in batch]
        max_cond_mel_len = max(cond_mel_lens)

        wav_lens = [x[3] for x in batch]
        max_wav_len = max(wav_lens)

        audio_lens = [x[4].shape[1] for x in batch]
        max_audio_len = max(audio_lens)

        mel_lens = [x[5].shape[1] for x in batch]
        max_mel_len = max(mel_lens)

        texts = []
        qmels = []
        cond_mels = []
        wavs = []
        mels = []

        for b in batch:
            text, qmel, cond_mel, wav_length, wav, mel = b

            texts.append(F.pad(text, (0, max_text_len - len(text)), value=0))

            qmels.append(F.pad(qmel, (0, max_qmel_len - len(qmel)), value=0))

            cond_mels.append(F.pad(cond_mel, (0, max_cond_mel_len - cond_mel.shape[1]), value=0))

            wavs.append(F.pad(wav, (0, max_audio_len - wav.shape[1]), value=0))

            mels.append(F.pad(mel, (0, max_mel_len - mel.shape[1]), value=0))

        padded_quant_mel = torch.stack(qmels)
        padded_cond_mel = torch.stack(cond_mels)
        padded_texts = torch.stack(texts)
        padded_wavs = torch.stack(wavs)
        padded_mel = torch.stack(mels)
        return {
            'text_inputs': padded_texts,
            'text_lens': LongTensor(text_lens),
            'padded_quant_mel': padded_quant_mel,
            'qmel_lens': LongTensor(qmel_lens),
            'padded_cond_mel': padded_cond_mel,
            'cond_lens': LongTensor(cond_mel_lens),
            'wav_lens': LongTensor(wav_lens),
            'wav': padded_wavs,
            'padded_mel': padded_mel,
        }


class DvaeMelDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        print(config['vae_train'])
        self.path = config['gpt_train']['train_file']
        self.sample_rate = config['vae_train']['sample_rate']
        self.n_mels = config['vae_train']['n_mels']

        try:
            self.mel_fmax = config['vae_train']['mel_fmax']
        except:
            self.mel_fmax = None

        self.pad_to = config['vae_train']['pad_to_samples']
        self.squeeze = config['vae_train']['squeeze']
        self.inj = TortoiseMelSpectrogramInjector({'in': 'wav', 'out': 'mel'},
                                                  n_mel_channels=self.n_mels,
                                                  sampling_rate=self.sample_rate,
                                                  mel_fmax=self.mel_fmax)
        self.audiopath_and_text = parse_filelist(self.path)
        self.mel_path = config['gpt_train']['mel_dir']

    def __getitem__(self, index):
        audiopath_and_text = self.audiopath_and_text[index]
        wav_file, text = audiopath_and_text[0], audiopath_and_text[1]

        audio, sr = torchaudio.load(wav_file)

        if audio.shape[0] > 1:
            audio = audio[0].unsqueeze(0)
        audio = torchaudio.transforms.Resample(sr, self.sample_rate)(audio).cuda()

        base_name = Path(wav_file).stem

        if not os.path.exists(f'{self.mel_path}/{base_name}.mel.pth'):
            print(base_name)
            mel = self.inj({'wav': audio.unsqueeze(0)})['mel']
            torch.save(mel.cpu().detach(), f'{self.mel_path}/{base_name}.mel.pth')

        mel = torch.load(f'{self.mel_path}/{base_name}.mel.pth')

        if mel.shape[-1] >= self.pad_to:
            start = torch.randint(0, mel.shape[-1] - self.pad_to + 1, (1,))
            mel = mel[:, :, start:start + self.pad_to]
            mask = torch.zeros_like(mel)
        else:
            mask = torch.zeros_like(mel)
            padding_needed = self.pad_to - mel.shape[-1]
            mel = F.pad(mel, (0, padding_needed))
            mask = F.pad(mask, (0, padding_needed), value=1)
        assert mel.shape[-1] == self.pad_to
        if self.squeeze:
            mel = mel.squeeze()

        return mel

    def __len__(self):
        return len(self.audiopath_and_text)
