import json
import os
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from audio.injectors import TortoiseDiscreteTokenInjector, TorchMelSpectrogramInjector
from common.custom_dataset import GptDataset
from text.text_tokenizer import TextBpeTokenizer
from models.gpt.gpt import TortoiseVoice
# from text.voice_tokenizer import VoiceBpeTokenizer
from utils.utils import latest_checkpoint_path, oldest_checkpoint_path, plot_spectrogram_to_numpy, summarize


def warmup(step):
    return 1


class Trainer(object):
    def __init__(self, cfg_path='configs/config.json'):
        self.cfg = json.load(open(cfg_path))

        MEL_NORM_FILE = self.cfg['gpt_train']['mel_normal_file']

        self.mel_path = Path(self.cfg['gpt_train']['mel_dir'])
        self.mel_path.mkdir(exist_ok=True, parents=True)

        self.logs_folder = Path(self.cfg['gpt_train']['logs_dir'])
        self.logs_folder.mkdir(exist_ok=True, parents=True)

        self.inj = TorchMelSpectrogramInjector({'in': 'wav', 'out': 'mel'}, power=1)  # power=1 for visual mel-spec

        # TODO: try using my own TextBpeTokenizer based on cmudict
        self.tokenizer = TextBpeTokenizer()

        self.gpt = TortoiseVoice(
            model_dim=1024,
            spec_dim=80,
            max_conditioning_inputs=2,
            max_mel_tokens=604,
            max_text_tokens=402,
            heads=16,
            layers=15,
            number_text_tokens=self.tokenizer.vocab_size(),
            start_text_token=self.tokenizer.vocab_size(),
            train_solo_embeddings=False,
            use_mel_codes_as_input=True,
            hifigan_in_sample_rate=22050
        )

        hifigan_checkpoint = torch.load('original_models/hifigan_decoder.pth', map_location=torch.device("cpu"))
        self.gpt.hifigan_decoder.load_state_dict(hifigan_checkpoint["model"], strict=True)

        # dvae_model_path = "original_models/dvae.pth"
        dvae_model_path = latest_checkpoint_path(Path(self.cfg['vae_train']['logs_dir']), f"dvae_[0-9]*")
        print(f'DVAE restored from {dvae_model_path}')

        self.code_inj = TortoiseDiscreteTokenInjector({'in': 'mel', 'out': 'codes'}, dvae_model_path)

        self.dataset = GptDataset(self.cfg, self.tokenizer, dvae_model_path, self.mel_path,
                                  mel_norm_file=MEL_NORM_FILE, is_eval=False)

        self.eval_dataset = GptDataset(self.cfg, self.tokenizer, dvae_model_path, self.mel_path,
                                       mel_norm_file=MEL_NORM_FILE, is_eval=True)

        self.dataloader = DataLoader(dataset=self.dataset, batch_size=self.cfg['gpt_dataloader']['batch_size'],
                                     collate_fn=self.dataset.collate_fn,
                                     drop_last=self.cfg['gpt_dataloader']['drop_last'],
                                     num_workers=self.cfg['gpt_dataloader']['num_workers'],
                                     pin_memory=self.cfg['gpt_dataloader']['pin_memory'],
                                     shuffle=self.cfg['gpt_dataloader']['shuffle'])

        self.eval_dataloader = DataLoader(self.eval_dataset, batch_size=1, shuffle=False, num_workers=0,
                                          pin_memory=False, collate_fn=self.eval_dataset.collate_fn)

        self.total_epochs = self.cfg['gpt_train']['train_epochs']
        self.val_freq = self.cfg['gpt_train']['val_freq']
        self.save_freq = self.cfg['gpt_train']['save_freq']

        self.optimizer = AdamW(self.gpt.parameters(), lr=self.cfg['gpt_train']['lr'], betas=(0.9, 0.96),
                               weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=warmup)

        self.epoch = 0
        self.step = 1

        self.load()

        self.mel_loss_weight = self.cfg['gpt_train']['mel_weight']
        self.text_loss_weight = self.cfg['gpt_train']['text_weight']

        self.writer = SummaryWriter(log_dir=os.path.join(self.logs_folder))

    def save(self):
        data = {
            'step': self.step,
            'epoch': self.epoch,
            'model': self.gpt.state_dict(),
        }
        torch.save(data, f'{self.logs_folder}/GPTT_{self.step}.pth')
        print(f"Saved GPT model to {self.logs_folder}/GPTT_{self.step}.pth'")
        keep_ckpts = self.cfg['gpt_train']['keep_ckpts']
        old_ckpt = oldest_checkpoint_path(f"{self.logs_folder}", f"GPTT_[0-9]*", preserved=keep_ckpts)
        if os.path.exists(old_ckpt):
            print(f"Removed {old_ckpt}")
            os.remove(old_ckpt)

    def load(self):
        try:
            print("loading model...")
            model_path = latest_checkpoint_path(f"{self.logs_folder}", f"GPTT_[0-9]*")
            gpt_checkpoint = torch.load(model_path, map_location="cpu")
            if 'step' in gpt_checkpoint:
                self.step = gpt_checkpoint['step'] + 1
            if 'epoch' in gpt_checkpoint:
                self.epoch = gpt_checkpoint['epoch']
            if 'model' in gpt_checkpoint:
                gpt_checkpoint = gpt_checkpoint['model']
            self.gpt.load_state_dict(gpt_checkpoint, strict=False)
            print(f'GPT model restored from {model_path}')
        except Exception as e:
            print(e)

    def train(self):
        self.gpt.cuda()
        self.gpt.train()

        for self.epoch in range(self.epoch, self.total_epochs + 1):
            for idx, batch in enumerate(self.dataloader):
                start_time = time.time()
                loss_dict = {}

                padded_cond_mel = batch['padded_cond_mel'].cuda()
                text_input = batch['text_inputs'].cuda()
                text_lens = batch['text_lens'].cuda()
                padded_quant_mel = batch['padded_quant_mel'].cuda()
                wav_lens = batch['wav_lens'].cuda()

                loss_text, loss_mel, mel_logits = self.gpt(padded_cond_mel,
                                                           text_input,
                                                           text_lens,
                                                           padded_quant_mel,
                                                           wav_lens)

                loss = loss_text * self.text_loss_weight + loss_mel * self.mel_loss_weight
                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(self.gpt.parameters(), max_norm=1)

                self.optimizer.step()
                self.optimizer.zero_grad()

                loss_dict["loss_text_ce"] = loss_text * self.text_loss_weight
                loss_dict["loss_mel_ce"] = loss_mel * self.mel_loss_weight
                loss_dict["loss"] = loss_dict["loss_text_ce"] + loss_dict["loss_mel_ce"]
                lr = self.scheduler.get_last_lr()[0]

                t = time.time() - start_time
                if self.step % 50 == 0:
                    print()
                    print(f'[Epoch: {self.epoch} | '
                          f'Iteration: {idx + 1}/{len(self.dataloader)} | {100. * (idx + 1) / len(self.dataloader):.2f}% | '
                          f'{t:.2f}s/it]')
    
                    print(
                        'Step: {}, '
                        'loss_text_ce: {:.7f}, '
                        'loss_mel_ce: {:.7f}, '
                        'total_loss: {:.7f}, '
                        'grad_norm: {:.7f}, '
                        'lr: {:.7f}'
                        .format(
                            self.step, loss_dict["loss_text_ce"], loss_dict["loss_mel_ce"],
                            loss_dict["loss"],
                            grad_norm,
                            lr
                        )
                    )

                if self.step % self.val_freq == 0:
                    scalar_dict = {
                        "gpt/loss_mel": loss_mel * self.mel_loss_weight,
                        "gpt/loss_text": loss_text * self.text_loss_weight,
                        "gpt/total_loss": loss,
                        "gpt/grad_norm": grad_norm,
                        "gpt/lr": self.scheduler.get_last_lr()[0]
                    }

                    summarize(
                        writer=self.writer,
                        global_step=self.step,
                        scalars=scalar_dict
                    )

                if self.step % self.save_freq == 0:
                    self.save()
                    # Eval won't work on Kaggle, as there is a problem with displaying tensorboard
                    # self.eval_model()

                self.step += 1
                self.scheduler.step()
            self.epoch += 1
            print(f'EPOCH ====> {self.epoch}')
        self.step += 1
        self.save()
        print(f'Training complete with {self.step} steps.')

    def eval_model(self):
        print('Evaluating...')
        self.gpt.eval()
        self.gpt.post_init_gpt2_config()

        for idx, batch in enumerate(self.eval_dataloader):
            if idx == 1:
                break
            text_tokens = batch['text_inputs'].to('cuda')
            mel_padded = batch['padded_mel']
            cond_mel = batch['padded_cond_mel'].to('cuda')
            wav = batch['wav']

            """text = "This model sounds really good and above all, it's reasonably fast."
            seq = self.tokenizer.encode(text)
            text_tokens = torch.IntTensor(seq).unsqueeze(0).to('cuda')
            text_tokens = F.pad(text_tokens, (0, 1))  # This may not be necessary.
            text_tokens = text_tokens.to('cuda')"""
            with torch.no_grad():
                codes = self.gpt.inference_speech(cond_mel, text_tokens,
                                                  do_sample=True,
                                                  top_p=.8,
                                                  temperature=.8,
                                                  num_return_sequences=1,
                                                  length_penalty=1.0,
                                                  repetition_penalty=2.0,
                                                  max_generate_length=600)

                text_len = torch.tensor([text_tokens.shape[-1]], device=text_tokens.device)
                expected_output_len = torch.tensor(
                    [codes.shape[-1] * self.gpt.mel_length_compression], device=text_tokens.device
                )

                latent = self.gpt(cond_mel,
                                  text_tokens,
                                  text_len,
                                  codes,
                                  expected_output_len,
                                  return_latent=True)

            speaker_embedding = self.gpt.get_speaker_embedding(wav[0, :, :], 22050)
            wav_out = self.gpt.hifigan_decoder(latent, g=speaker_embedding).detach().cpu()
            hifi_mel = self.inj({'wav': wav_out})['mel']
            gt_mel = self.inj({'wav': wav[0, :, :]})['mel']

            with torch.no_grad():
                image_dict_gen = {
                    "gpt/gen_mel": plot_spectrogram_to_numpy(hifi_mel[0, :, :].detach().unsqueeze(-1).cpu()),
                }
                audios_dict_gen = {
                    "gpt/gen_audio": wav_out,
                }

                image_dict_gt = {
                    "gpt/gt_mel": plot_spectrogram_to_numpy(gt_mel[0, :, :].detach().unsqueeze(-1).cpu()),
                }
                audios_dict_gt = {
                    "gpt/gt_audio": wav[0, :, :].detach().cpu(),
                }

                summarize_gen = summarize
                summarize_gt = summarize

                summarize_gen(
                    writer=self.writer,
                    global_step=self.step,
                    images=image_dict_gen,
                    audios=audios_dict_gen,
                    audio_sampling_rate=22050
                )

                summarize_gt(
                    writer=self.writer,
                    global_step=self.step,
                    images=image_dict_gt,
                    audios=audios_dict_gt,
                    audio_sampling_rate=22050
                )

            try:
                del self.gpt.gpt.wte
            except Exception as e:
                print(f'1. ---> {e}')
            try:
                del self.gpt.inference_model
            except Exception as e:
                print(f'2. ---> {e}')

        self.gpt.train()


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
