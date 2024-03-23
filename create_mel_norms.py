import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from audio.injectors import TorchMelSpectrogramInjector, TorchMelSpectrogramInjector_Base
from common.custom_dataset import GptDataset
from common.custom_dataset import DvaeMelDataset
from text.text_tokenizer import TextBpeTokenizer
from ttts.gpt.voice_tokenizer import VoiceBpeTokenizer
from utils.utils import latest_checkpoint_path

cfg_path = 'configs/config.json'
cfg = json.load(open(cfg_path))

tokenizer = VoiceBpeTokenizer()
inj = TorchMelSpectrogramInjector_Base({'in': 'wav', 'out': 'mel'}).cuda()

dvae_model_path = latest_checkpoint_path(cfg['vae_train']['logs_dir'], f"dvae_[0-9]*")
print(f'Using DVAE model {dvae_model_path}...')

dataset = DvaeMelDataset(cfg)
dataloader = DataLoader(dataset=dataset, batch_size=4, drop_last=False, num_workers=0, pin_memory=False, shuffle=False)

mels = []
for batch in tqdm(dataloader):
    mel = batch.to('cuda').squeeze(1)
    mels.append(mel.mean((0, 2)).cpu())
    if len(mels) > 10:
        break
mel_norms = torch.stack(mels).mean(0)
torch.save(mel_norms, 'custom_models/mel_norms.pth')
