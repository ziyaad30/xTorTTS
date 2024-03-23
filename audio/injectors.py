import torch
import torchaudio
from tortoise.dvae import DiscreteVAE


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


class TorchMelSpectrogramInjector(Injector):
    def __init__(self, opt, filter_length=1024, hop_length=256, win_length=1024, n_mel_channels=80, mel_fmin=0,
                 mel_fmax=8000, sampling_rate=22050, normalize=False, mel_norm_file=None, power=2):
        super().__init__(opt)
        # These are the default tacotron values for the MEL spectrogram.
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.sampling_rate = sampling_rate
        self.power = power
        self.mel_stft = torchaudio.transforms.MelSpectrogram(n_fft=self.filter_length, hop_length=self.hop_length,
                                                             win_length=self.win_length, power=self.power,
                                                             normalized=normalize, sample_rate=self.sampling_rate,
                                                             f_min=self.mel_fmin, f_max=self.mel_fmax,
                                                             n_mels=self.n_mel_channels, norm="slaney")
        self.mel_norm_file = mel_norm_file
        if self.mel_norm_file is not None:
            self.mel_norms = torch.load(self.mel_norm_file)
        else:
            self.mel_norms = None

    def forward(self, state):
        inp = state[self.input]
        if (
                len(inp.shape) == 3
        ):  # Automatically squeeze out the channels dimension if it is present (assuming mono-audio)
            inp = inp.squeeze(1)
        assert len(inp.shape) == 2
        self.mel_stft = self.mel_stft.to(inp.device)
        mel = self.mel_stft(inp)
        # Perform dynamic range compression
        mel = torch.log(torch.clamp(mel, min=1e-5))
        if self.mel_norms is not None:
            self.mel_norms = self.mel_norms.to(mel.device)
            mel = mel / self.mel_norms.unsqueeze(0).unsqueeze(-1)
        return {self.output: mel}


class TorchMelSpectrogramInjector_Base(Injector):
    # this injector is used when training DVAE and generating the mel_norms_file
    def __init__(self, opt, filter_length=1024, hop_length=256, win_length=1024, n_mel_channels=80, mel_fmin=0,
                 mel_fmax=8000, sampling_rate=22050, normalize=False):
        super().__init__(opt)
        # These are the default tacotron values for the MEL spectrogram.
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.sampling_rate = sampling_rate
        self.normalize = normalize
        self.mel_stft = torchaudio.transforms.MelSpectrogram(n_fft=self.filter_length, hop_length=self.hop_length,
                                                             win_length=self.win_length, power=2,
                                                             normalized=self.normalize, sample_rate=self.sampling_rate,
                                                             f_min=self.mel_fmin, f_max=self.mel_fmax,
                                                             n_mels=self.n_mel_channels)

    def forward(self, state):
        inp = state[self.input]
        if (
                len(inp.shape) == 3
        ):  # Automatically squeeze out the channels dimension if it is present (assuming mono-audio)
            inp = inp.squeeze(1)
        assert len(inp.shape) == 2
        self.mel_stft = self.mel_stft.to(inp.device)
        mel = self.mel_stft(inp)
        # Perform dynamic range compression
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return {self.output: mel}


class TortoiseDiscreteTokenInjector(Injector):
    def __init__(self, opt, path, channels=80):
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
        self.dvae.load_state_dict(dvae_checkpoint, strict=True)

    def forward(self, state):
        inp = state[self.input]
        with torch.no_grad():
            self.dvae = self.dvae.to(inp.device)
            codes = self.dvae.get_codebook_indices(inp)
            return {self.output: codes}
