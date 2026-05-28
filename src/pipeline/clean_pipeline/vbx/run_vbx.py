"""
VBx: Variational Bayes HMM Clustering for speaker diarization.
Uses pretrained ResNet101_16kHz ONNX x-vector extractor + Kaldi PLDA.
"""

import os
import sys
import shutil
import numpy as np
import h5py
import soundfile as sf
from scipy.linalg import eigh
from scipy.special import softmax
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage as scipy_linkage, fcluster

_VBX_DIR = os.path.dirname(os.path.abspath(__file__))
_CLEAN_PIPELINE_DIR = os.path.dirname(_VBX_DIR)
_ONNX_PATH = os.path.join(_VBX_DIR, "models", "ResNet101_16kHz", "nnet", "final.onnx")
_PLDA_PATH = os.path.join(_VBX_DIR, "models", "ResNet101_16kHz", "plda")
_XFORM_PATH = os.path.join(_VBX_DIR, "models", "ResNet101_16kHz", "transform.h5")

for _p in [_VBX_DIR, _CLEAN_PIPELINE_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from features import mel_fbank_mx, fbank_htk, cmvn_floating_kaldi, add_dither, povey_window
from diarization_lib import l2_norm, cos_similarity, twoGMMcalib_lin
from kaldi_utils import read_plda
from VBx import VBx

from clean import parse_diarization, write_diarization, merge_same_speaker_segments

# Feature extraction constants (16kHz, matching VBx training conditions)
_SR = 16000
_WINLEN = 400     # 25ms at 16kHz
_NOVERLAP = 240   # 15ms overlap → 10ms hop
_LC = 150
_RC = 149
_SEG_LEN = 144    # frames per x-vector window
_SEG_JUMP = 24    # frame jump between windows


def _load_onnx_model():
    import onnxruntime
    sess = onnxruntime.InferenceSession(_ONNX_PATH)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    return sess, input_name, label_name


def _extract_segment_xvector(signal, start_sample, end_sample, window, fbank_mx_arr, model_tuple):
    """Extract one x-vector per diarization segment (average over sliding windows)."""
    sess, input_name, label_name = model_tuple
    seg = signal[start_sample:end_sample]
    if len(seg) < int(0.01 * _SR):
        return None

    # Mirror edges to match VBx reference feature extraction
    seg = np.r_[seg[_NOVERLAP // 2 - 1::-1], seg, seg[-1:-_WINLEN // 2 - 1:-1]]
    fea = fbank_htk(seg, window, _NOVERLAP, fbank_mx_arr, USEPOWER=True, ZMEANSOURCE=True)
    fea = cmvn_floating_kaldi(fea, _LC, _RC, norm_vars=False).astype(np.float32)

    slen = len(fea)
    xvectors = []
    start = -_SEG_JUMP
    for start in range(0, slen - _SEG_LEN, _SEG_JUMP):
        data = fea[start:start + _SEG_LEN]
        xv = sess.run([label_name], {input_name: data.T[np.newaxis, :, :]})[0].squeeze()
        if not np.isnan(xv).any():
            xvectors.append(xv)

    # Last (possibly shorter) block
    tail_start = start + _SEG_JUMP
    if slen - tail_start >= 10:
        data = fea[tail_start:slen]
        xv = sess.run([label_name], {input_name: data.T[np.newaxis, :, :]})[0].squeeze()
        if not np.isnan(xv).any():
            xvectors.append(xv)

    if not xvectors:
        if slen >= 10:
            xv = sess.run([label_name], {input_name: fea[:slen].T[np.newaxis, :, :]})[0].squeeze()
            if not np.isnan(xv).any():
                return xv.astype(float)
        return None

    return np.mean(np.stack(xvectors), axis=0).astype(float)


def _ahc_init(x, max_speakers, init_smoothing):
    """AHC-based soft initialization for VBx gamma."""
    n = len(x)
    if n == 1:
        return np.ones((1, 1))

    scr_mx = cos_similarity(x)
    thr, _ = twoGMMcalib_lin(scr_mx.ravel())

    dist_condensed = squareform(-scr_mx, checks=False)
    lin_mat = scipy_linkage(dist_condensed, method='average')
    adjust = abs(lin_mat[:, 2].min())
    lin_mat[:, 2] += adjust
    labels1st = fcluster(lin_mat, -(thr + 0.0) + adjust, criterion='distance') - 1

    n_spk = min(int(np.max(labels1st)) + 1, max_speakers)
    qinit = np.zeros((n, n_spk))
    qinit[range(n), np.minimum(labels1st, n_spk - 1)] = 1.0
    qinit = softmax(qinit * init_smoothing, axis=1)
    return qinit


def run_vbx(args) -> str:
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "cleansed_diarization.txt")

    print("\n[VBx] Step 1: Loading diarization and audio")
    segments = parse_diarization(args.diarization_path)
    if not segments:
        write_diarization(output_path, [])
        return output_path

    signal, sr = sf.read(args.audio_path)
    if signal.ndim > 1:
        signal = signal[:, 0]
    if sr != _SR:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, _SR)
        signal = resample_poly(signal, _SR // g, sr // g)

    # Scale to 16-bit integer range and add dither (matching VBx reference)
    signal = add_dither((signal * 2**15).astype(int))

    print("\n[VBx] Step 2: Loading ONNX model")
    model_tuple = _load_onnx_model()

    window = povey_window(_WINLEN)
    fbank_mx_arr = mel_fbank_mx(_WINLEN, _SR, NUMCHANS=64, LOFREQ=20.0, HIFREQ=7600, htk_bug=False)

    print("\n[VBx] Step 3: Extracting x-vectors")
    xvectors, valid_segs = [], []
    for seg in segments:
        s, e = int(seg.start * _SR), int(seg.end * _SR)
        xv = _extract_segment_xvector(signal, s, e, window, fbank_mx_arr, model_tuple)
        if xv is not None:
            xvectors.append(xv)
            valid_segs.append(seg)

    print(f"  Extracted {len(xvectors)} x-vectors from {len(segments)} segments")

    if len(xvectors) < 2:
        shutil.copy(args.diarization_path, output_path)
        print("[VBx] Too few segments, copying input to output")
        return output_path

    X = np.stack(xvectors)  # (N, 256)

    print("\n[VBx] Step 4: Loading PLDA and LDA transform")
    plda_mu, plda_tr, plda_psi = read_plda(_PLDA_PATH)

    with h5py.File(_XFORM_PATH, 'r') as f:
        mean1 = np.array(f['mean1'])
        mean2 = np.array(f['mean2'])
        lda = np.array(f['lda'])

    # Apply two-stage LDA transform (exact match to vbhmm.py reference)
    x_lda = l2_norm(lda.T.dot((l2_norm(X - mean1)).T).T - mean2)

    # PLDA eigendecomposition (from vbhmm.py)
    W = np.linalg.inv(plda_tr.T.dot(plda_tr))
    B = np.linalg.inv((plda_tr.T / plda_psi).dot(plda_tr))
    acvar, wccn = eigh(B, W)
    plda_psi_new = acvar[::-1]
    plda_tr_new = wccn.T[::-1]

    lda_dim = args.vbx_lda_dim
    fea = (x_lda - plda_mu).dot(plda_tr_new.T)[:, :lda_dim]

    print("\n[VBx] Step 5: AHC initialization")
    qinit = _ahc_init(x_lda, args.vbx_max_speakers, args.vbx_init_smoothing)

    print("\n[VBx] Step 6: Running VBx inference")
    gamma, pi, Li = VBx(
        fea, plda_psi_new[:lda_dim],
        loopProb=args.vbx_loop_prob,
        Fa=args.vbx_fa,
        Fb=args.vbx_fb,
        pi=qinit.shape[1],
        gamma=qinit,
        maxIters=40,
        epsilon=1e-6,
    )

    labels = np.argmax(gamma, axis=1)
    n_active = int(np.sum(pi > 0.01))
    print(f"  VBx found {n_active} active speakers")

    print("\n[VBx] Step 7: Writing output")
    out_segs = [
        {"start": s.start, "end": s.end, "speaker": f"SPEAKER_{l:02d}"}
        for s, l in zip(valid_segs, labels)
    ]
    out_segs.sort(key=lambda x: x["start"])
    out_segs = merge_same_speaker_segments(out_segs)

    write_diarization(output_path, out_segs)
    print(f"[VBx] Done. {len(segments)} → {len(out_segs)} segments. Output: {output_path}")
    return output_path
