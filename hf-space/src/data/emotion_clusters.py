"""
Emotion label mapping and clinical cluster definitions.
Implements Equation (1): C(y) cluster mapping from 27 GoEmotions labels.
"""

from typing import Dict, List

# ── 27 GoEmotions labels → 4 clinical clusters ─────────────────────────────────
EMOTION_TO_CLUSTER: Dict[str, str] = {
    # Distress cluster
    "grief": "Distress",
    "nervousness": "Distress",
    "fear": "Distress",
    "sadness": "Distress",
    "remorse": "Distress",
    # Joy / Resilience cluster
    "joy": "Resilience",
    "admiration": "Resilience",
    "excitement": "Resilience",
    "relief": "Resilience",
    "amusement": "Resilience",
    "approval": "Resilience",
    "curiosity": "Resilience",
    "desire": "Resilience",
    "gratitude": "Resilience",
    "love": "Resilience",
    "optimism": "Resilience",
    "pride": "Resilience",
    "realization": "Resilience",
    # Anger / Aggression cluster
    "anger": "Aggression",
    "annoyance": "Aggression",
    "disgust": "Aggression",
    "disapproval": "Aggression",
    "embarrassment": "Aggression",
    # Ambiguity / Neutral cluster
    "confusion": "Ambiguity",
    "disappointment": "Ambiguity",
    "surprise": "Ambiguity",
    "caring": "Ambiguity",
    "neutral": "Ambiguity",
}

CLUSTER_LABELS: List[str] = ["Distress", "Resilience", "Aggression", "Ambiguity"]
CLUSTER_TO_IDX: Dict[str, int] = {label: i for i, label in enumerate(CLUSTER_LABELS)}
IDX_TO_CLUSTER: Dict[int, str] = {i: label for label, i in CLUSTER_TO_IDX.items()}

NUM_CLUSTERS = len(CLUSTER_LABELS)

# RAVDESS emotion code → cluster
RAVDESS_LABEL_MAP: Dict[int, str] = {
    1: "Ambiguity",   # neutral
    2: "Ambiguity",   # calm
    3: "Resilience",  # happy
    4: "Distress",    # sad
    5: "Aggression",  # angry
    6: "Distress",    # fearful
    7: "Aggression",  # disgust
    8: "Ambiguity",   # surprised
}

# IEMOCAP label → cluster
IEMOCAP_LABEL_MAP: Dict[str, str] = {
    "neu": "Ambiguity",
    "hap": "Resilience",
    "exc": "Resilience",
    "sad": "Distress",
    "ang": "Aggression",
    "fru": "Aggression",
    "fea": "Distress",
    "sur": "Ambiguity",
    "dis": "Aggression",
    "oth": "Ambiguity",
}


def map_go_emotions_label(raw_label: str) -> str:
    """Map a raw GoEmotions label string to a clinical cluster name."""
    return EMOTION_TO_CLUSTER.get(raw_label.lower(), "Ambiguity")


def map_go_emotions_to_idx(raw_label: str) -> int:
    """Map a raw GoEmotions label to cluster index (0–3)."""
    cluster = map_go_emotions_label(raw_label)
    return CLUSTER_TO_IDX[cluster]


def map_ravdess_label(emotion_code: int) -> str:
    """Map RAVDESS numeric emotion code to cluster name."""
    return RAVDESS_LABEL_MAP.get(emotion_code, "Ambiguity")


def map_iemocap_label(label: str) -> str:
    """Map IEMOCAP string label to cluster name."""
    return IEMOCAP_LABEL_MAP.get(label.lower(), "Ambiguity")
