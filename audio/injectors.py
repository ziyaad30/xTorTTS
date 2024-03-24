import torch
import torchaudio
from models.dvae.dvae import DiscreteVAE


def safe_log(x: torch.Tensor, clip_val: float = 1e-7) -> torch.Tensor:
    """
    Computes the element-wise logarithm of the input tensor with clipping to avoid near-zero values.

    Args:
        x (Tensor): Input tensor.
        clip_val (float, optional): Minimum value to clip the input tensor. Defaults to 1e-7.

    Returns:
        Tensor: Element-wise logarithm of the input tensor with clipping applied.
    """
    return torch.log(torch.clip(x, min=clip_val))


class Injector(torch.nn.Module):
    def __init__(self, opt):
        super(Injector, self).__init__()

        if 'in' in opt.keys():
            self.input = opt['in']
        if 'out' in opt.keys():
            self.output = opt['out']

    # This should return a dict of new state variables.
    def forward(self, state):
        raise NotImplementedError


class TortoiseMelSpectrogramInjector(Injector):
    def __init__(self, opt, n_mel_channels=100, sampling_rate=24000, n_fft=1024, hop_length=256, win_length=None,
                 mel_fmin=0, mel_fmax=None, normalized=False, padding="center", power=1):
        super().__init__(opt)
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mel_channels
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.sampling_rate = sampling_rate
        self.power = power
        self.normalized = normalized
        self.mel_stft = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sampling_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            power=self.power,
            normalized=self.normalized,
            f_min=self.mel_fmin,
            f_max=self.mel_fmax,
            n_mels=self.n_mels,
            center=padding == "center",
        )

    def forward(self, state):
        with torch.no_grad():
            inp = state[self.input]
            if self.padding == "same":
                pad = self.mel_stft.win_length - self.mel_stft.hop_length
                inp = torch.nn.functional.pad(inp, (pad // 2, pad // 2), mode="reflect")
            if len(inp.shape) == 3:  # Automatically squeeze out the channels dimension if it is present (assuming mono-audio)
                inp = inp.squeeze(1)
            assert len(inp.shape) == 2
            self.mel_stft = self.mel_stft.to(inp.device)
            mel = self.mel_stft(inp)
            mel = safe_log(mel)
            return {self.output: mel}


class TortoiseDiscreteTokenInjector(Injector):
    def __init__(self, opt, path, channels=100, strict=True):
        super().__init__(opt)
        self.channels = channels
        self.dvae = DiscreteVAE(channels=self.channels,
                                normalization=None,
                                positional_dims=1,
                                num_tokens=8192,
                                codebook_dim=512,
                                hidden_dim=512,
                                num_resnet_blocks=3,
                                kernel_size=3,
                                num_layers=2,
                                use_transposed_convs=False)

        self.dvae.eval()
        dvae_checkpoint = torch.load(path, map_location=torch.device("cpu"))
        if 'model' in dvae_checkpoint:
            dvae_checkpoint = dvae_checkpoint['model']
        self.dvae.load_state_dict(dvae_checkpoint, strict=strict)

    def forward(self, state):
        inp = state[self.input]
        with torch.no_grad():
            self.dvae = self.dvae.to(inp.device)
            codes = self.dvae.get_codebook_indices(inp)
            return {self.output: codes}


class Codes2MelInjector(Injector):
    def __init__(self, opt, path, channels=100, strict=True):
        super().__init__(opt)
        self.channels = channels
        self.dvae = DiscreteVAE(channels=self.channels,
                                normalization=None,
                                positional_dims=1,
                                num_tokens=8192,
                                codebook_dim=512,
                                hidden_dim=512,
                                num_resnet_blocks=3,
                                kernel_size=3,
                                num_layers=2,
                                use_transposed_convs=False)

        self.dvae.eval()
        dvae_checkpoint = torch.load(path, map_location=torch.device("cpu"))
        if 'model' in dvae_checkpoint:
            dvae_checkpoint = dvae_checkpoint['model']
        self.dvae.load_state_dict(dvae_checkpoint, strict=strict)

    def forward(self, state):
        inp = state[self.input]
        with torch.no_grad():
            self.dvae = self.dvae.to(inp.device)
            dvae_mel = self.dvae.decode(inp)
            return {self.output: dvae_mel}
