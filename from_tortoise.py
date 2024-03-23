import json
from pathlib import Path
import torch
from utils.io import load_fsspec


cfg_path = 'configs/config.json'
cfg = json.load(open(cfg_path))
logs_folder = Path(cfg['gpt_train']['logs_dir'])
logs_folder.mkdir(exist_ok=True, parents=True)

checkpoint = load_fsspec("original_models/autoregressive.pth", map_location=torch.device("cpu"))

data = {
    'step': 0,
    'epoch': 0,
    'model': checkpoint,
}
torch.save(data, f'{logs_folder}/GPTT_0.pth')

