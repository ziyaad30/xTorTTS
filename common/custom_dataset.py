import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from torch import LongTensor
from torch.utils.data import Dataset

from audio.injectors import TortoiseDiscreteTokenInjector, TorchMelSpectrogramInjector, TorchMelSpectrogramInjector_Base


def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split_char) for line in f]
    return filepaths_and_text


class GptDataset(Dataset):
    def __init__(self, config, tokenizer, dvae_path, mel_path, mel_norm_file=None, is_eval=False):
        self.is_eval = is_eval
        if not self.is_eval:
            self.path = config['gpt_train']['train_file']
        else:
            self.path = config['gpt_train']['valid_file']
        self.tokenizer = tokenizer
        self.audiopath_and_text = parse_filelist(self.path)

        self.mel_norm_file = mel_norm_file
        self.inj = TorchMelSpectrogramInjector({'in': 'wav', 'out': 'mel'}, mel_norm_file=self.mel_norm_file)
        self.mel_path = mel_path
        self.dvae_path = dvae_path
        self.code_inj = TortoiseDiscreteTokenInjector({'in': 'mel', 'out': 'codes'}, self.dvae_path)

    def get_text(self, text):
        tokens = self.tokenizer.encode(text)
        tokens = torch.IntTensor(tokens)
        assert not torch.any(tokens == 1), f"UNK token found in {text} -> {self.tokenizer.decode(tokens)}"
        # The stop token should always be sacred.
        assert not torch.any(tokens == 0), f"Stop token found in {text}"
        return tokens

    def __getitem__(self, index):
        try:
            audiopath_and_text = self.audiopath_and_text[index]
            wav_file, text = audiopath_and_text[0], audiopath_and_text[1]

            tseq = self.get_text(text)

            audio, sr = torchaudio.load(wav_file)

            if audio.shape[0] > 1:
                audio = audio[0].unsqueeze(0)
            audio = torchaudio.transforms.Resample(sr, 22050)(audio).cuda()

            base_name = Path(wav_file).stem

            if not os.path.exists(f'{self.mel_path}/{base_name}.mel.pth'):
                mel = self.inj({'wav': audio.unsqueeze(0)})['mel']
                torch.save(mel.cpu().detach(), f'{self.mel_path}/{base_name}.mel.pth')

            mel = torch.load(f'{self.mel_path}/{base_name}.mel.pth')

            if not os.path.exists(f'{self.mel_path}/{base_name}.melvq.pth'):
                # print(base_name)
                code = self.code_inj({'mel': mel.to('cuda')})['codes']
                torch.save(code, f'{self.mel_path}/{base_name}.melvq.pth')

            qmel = torch.load(f'{self.mel_path}/{base_name}.melvq.pth')

            mel = mel[0]
            qmel = qmel[0]
            wav_length = mel.shape[1] * 256

            # print(base_name)
            # print(qmel.shape[0])

            split = random.randint(int(mel.shape[1] // 3), int(mel.shape[1] // 3 * 2))
            if random.random() > 0.5:
                cond_mel = mel[:, :split]
            else:
                cond_mel = mel[:, split:]
        except:
            return None

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
        self.path = config['gpt_train']['train_file']
        self.sample_rate = config['vae_train']['sample_rate']
        self.n_mels = config['vae_train']['n_mels']

        self.pad_to = config['vae_train']['pad_to_samples']
        self.squeeze = config['vae_train']['squeeze']
        self.inj = TorchMelSpectrogramInjector_Base({'in': 'wav', 'out': 'mel'})
        self.audiopath_and_text = parse_filelist(self.path)

    def __getitem__(self, index):
        audiopath_and_text = self.audiopath_and_text[index]
        wav_file, text = audiopath_and_text[0], audiopath_and_text[1]

        audio, sr = torchaudio.load(wav_file)

        if audio.shape[0] > 1:
            audio = audio[0].unsqueeze(0)
        audio = torchaudio.transforms.Resample(sr, self.sample_rate)(audio).cuda()

        mel = self.inj({'wav': audio.unsqueeze(0)})['mel']

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
