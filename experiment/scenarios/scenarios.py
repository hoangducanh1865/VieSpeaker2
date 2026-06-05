"""Curated scenario matrix for VieSpeaker2 benchmarking.

Each scenario is a dict describing one configuration to evaluate on all samples.
The runner (run_scenarios.py) precomputes shared P1/P2 outputs once and then
materialises each scenario's final hypothesis cheaply.

Fields:
    id        unique short slug (also the output sub-folder name)
    desc      human description (shown in the report)
    kind      "p1" | "p2" | "p3" | "fusion"
    p1        which P1 model to use as base/hypothesis: "cloud" | "local"
    asd       (p2/p3-on-p2) ASD model: "lr_asd" | "loconet"
    stage_in  (p3 only) what to cleanse: "p1" (clean P1 output) | "p2" (clean P2 output)
    method    (p3 only) ahc | cdgcn | vbx | dover-lap | nme-sc
    embedding (p3 only) ecapa | wespeaker34 | wespeaker293 | campplus | redimnet
    fusion    (fusion only) list of input specs, each "p1:cloud" | "p1:local" | "p2:lr_asd"
    best_effort  if True, failure is expected-tolerable (needs network / extra model)
"""

# P1 model id mapping
P1_MODELS = {
    "cloud": "pyannote/speaker-diarization-precision-2",
    "local": "pyannote/speaker-diarization-3.1",
}

# The P1 model that P2/P3/fusion build on by default.
BASE_P1 = "cloud"

ALL_SAMPLES = ["drama", "interview_clean", "interview_noise", "movie", "sample_0", "singing"]

# A light subset used by --smoke (fast sanity check).
SMOKE_SAMPLES = ["interview_clean", "interview_noise"]
SMOKE_SCENARIOS = ["p1_cloud", "p1_local", "p3_ahc_ecapa", "fusion_p1_p2"]


SCENARIOS = [
    # ---- P1 baselines ------------------------------------------------------
    {"id": "p1_cloud", "kind": "p1", "p1": "cloud",
     "desc": "P1 audio-only — pyannote precision-2 (cloud)"},
    {"id": "p1_local", "kind": "p1", "p1": "local",
     "desc": "P1 audio-only — pyannote speaker-diarization-3.1 (local)"},

    # ---- P3 cleansing applied to the P1 (cloud) hypothesis -----------------
    {"id": "p3_ahc_ecapa", "kind": "p3", "stage_in": "p1", "p1": "cloud",
     "method": "ahc", "embedding": "ecapa",
     "desc": "P1+P3 AHC, ECAPA-TDNN embeddings (VoxCeleb EN)"},
    {"id": "p3_ahc_wespeaker34", "kind": "p3", "stage_in": "p1", "p1": "cloud",
     "method": "ahc", "embedding": "wespeaker34",
     "desc": "P1+P3 AHC, WeSpeaker ResNet34 embeddings"},
    {"id": "p3_ahc_campplus", "kind": "p3", "stage_in": "p1", "p1": "cloud",
     "method": "ahc", "embedding": "campplus",
     "desc": "P1+P3 AHC, CAM++ embeddings (zh-cn)"},
    {"id": "p3_ahc_redimnet", "kind": "p3", "stage_in": "p1", "p1": "cloud",
     "method": "ahc", "embedding": "redimnet", "best_effort": True,
     "desc": "P1+P3 AHC, ReDimNet-B6 embeddings (torch.hub; best-effort)"},
    {"id": "p3_cdgcn_ecapa", "kind": "p3", "stage_in": "p1", "p1": "cloud",
     "method": "cdgcn", "embedding": "ecapa",
     "desc": "P1+P3 CDGCN (KNN graph + Leiden), ECAPA embeddings"},
    {"id": "p3_vbx", "kind": "p3", "stage_in": "p1", "p1": "cloud",
     "method": "vbx", "embedding": "ecapa",
     "desc": "P1+P3 VBx (Bayesian HMM + PLDA, ResNet101 x-vectors)"},
    {"id": "p3_nmesc_ecapa", "kind": "p3", "stage_in": "p1", "p1": "cloud",
     "method": "nme-sc", "embedding": "ecapa",
     "desc": "P1+P3 NME-SC (auto-tuning spectral clustering), ECAPA"},
    {"id": "p3_nmesc_wespeaker293", "kind": "p3", "stage_in": "p1", "p1": "cloud",
     "method": "nme-sc", "embedding": "wespeaker293",
     "desc": "P1+P3 NME-SC, WeSpeaker ResNet293 embeddings"},

    # ---- P2 audio-visual (redesigned, relabel-only) -----------------------
    {"id": "p2_lrasd", "kind": "p2", "p1": "cloud", "asd": "lr_asd",
     "desc": "P1+P2 audio-visual supplement (LR-ASD), relabel-only"},
    {"id": "p2_loconet", "kind": "p2", "p1": "cloud", "asd": "loconet", "best_effort": True,
     "desc": "P1+P2 audio-visual supplement (LoCoNet), relabel-only (best-effort)"},

    # ---- P2 then P3 -------------------------------------------------------
    {"id": "p2p3_ahc_ecapa", "kind": "p3", "stage_in": "p2", "p1": "cloud", "asd": "lr_asd",
     "method": "ahc", "embedding": "ecapa",
     "desc": "P1+P2(LR-ASD)+P3 AHC, ECAPA embeddings"},

    # ---- Fusion (DOVER-Lap) ----------------------------------------------
    {"id": "fusion_p1_p2", "kind": "fusion", "asd": "lr_asd",
     "fusion": ["p1:cloud", "p2:lr_asd"],
     "desc": "Fusion: P1 (cloud) + P2 (LR-ASD) via DOVER-Lap"},
    {"id": "fusion_cloud_local_p2", "kind": "fusion", "asd": "lr_asd",
     "fusion": ["p1:cloud", "p1:local", "p2:lr_asd"],
     "desc": "Fusion: P1 cloud + P1 local + P2 (LR-ASD) via DOVER-Lap"},
]


def get_scenarios(only=None, smoke=False):
    scen = SCENARIOS
    if smoke:
        scen = [s for s in scen if s["id"] in SMOKE_SCENARIOS]
    if only:
        only_set = set(only)
        scen = [s for s in scen if s["id"] in only_set]
    return scen


def required_p1_keys(scenarios):
    """Set of P1 model keys ('cloud'/'local') that must be precomputed."""
    keys = set()
    for s in scenarios:
        if s["kind"] == "p1":
            keys.add(s["p1"])
        elif s["kind"] in ("p2", "p3"):
            keys.add(s.get("p1", BASE_P1))
        elif s["kind"] == "fusion":
            for spec in s["fusion"]:
                tag, val = spec.split(":")
                if tag == "p1":
                    keys.add(val)
                elif tag == "p2":
                    keys.add(BASE_P1)  # P2 is built on the base P1
    return keys


def required_p2_keys(scenarios):
    """Set of (p1_key, asd) pairs whose P2 face pipeline must be precomputed."""
    keys = set()
    for s in scenarios:
        if s["kind"] == "p2":
            keys.add((s.get("p1", BASE_P1), s["asd"]))
        elif s["kind"] == "p3" and s.get("stage_in") == "p2":
            keys.add((s.get("p1", BASE_P1), s["asd"]))
        elif s["kind"] == "fusion":
            for spec in s["fusion"]:
                tag, val = spec.split(":")
                if tag == "p2":
                    keys.add((BASE_P1, val))
    return keys
