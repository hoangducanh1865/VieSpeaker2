import os
import torch
import torchaudio.compliance.kaldi as kaldi
from .ecapa_tdnn import ECAPA_TDNN_GLOB_c1024

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "avg_model.pt")


def apply_cmvn(feats, norm_mean=True, norm_var=False):
    if norm_mean:
        feats = feats - torch.mean(feats, dim=1, keepdim=True)
    if norm_var:
        feats = feats / torch.sqrt(torch.var(feats, dim=1, keepdim=True) + 1e-7)
    return feats


def compute_fbank(waveform):
    waveform = waveform * (1 << 15)
    mat = kaldi.fbank(
        waveform,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        sample_frequency=16000,
        window_type="hamming",
        use_energy=False,
    )
    return mat


def get_pretrained_model(ckp_path=MODEL_PATH):
    model = ECAPA_TDNN_GLOB_c1024(feat_dim=80, embed_dim=192, pooling_func="ASTP")
    state_dict = torch.load(ckp_path, map_location="cpu")
    state_dict.pop("projection.weight", None)
    state_dict.pop("projection.bias", None)
    model.load_state_dict(state_dict)
    return model
