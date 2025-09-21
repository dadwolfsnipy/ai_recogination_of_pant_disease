import os
import random
from typing import List

_TIP_BANK = {
    "tomato": [
        "Stake or cage plants to improve airflow and reduce blight risk.",
        "Use calcium-rich amendments (e.g., crushed eggshells) to reduce blossom end rot.",
        "Mulch to conserve moisture and keep soil splash off leaves.",
    ],
    "pepper": [
        "Keep soil consistently moist; avoid sudden dry–wet swings to prevent flower drop.",
        "Pinch early flowers to encourage stronger vegetative growth.",
        "Watch for aphids on new growth; a mild soap spray can help.",
    ],
    "potato": [
        "Hill soil around stems to protect developing tubers from sunlight.",
        "Rotate away from other solanaceae to reduce late blight pressure.",
        "Remove lower leaves with lesions early to slow spread.",
    ],
    "cabbage": [
        "Use row covers early to keep cabbage moths from laying eggs.",
        "Maintain even moisture; stress can cause splitting.",
        "Inspect for clustered eggs under leaves and remove by hand.",
    ],
}

_GENERAL = [
    "Water in the morning to limit fungal growth overnight.",
    "Prefer deep, infrequent watering over frequent shallow watering.",
    "Maintain 30–60 cm spacing (depending on crop) for airflow.",
    "Apply balanced N–P–K fertilizer based on a simple soil test.",
    "Rotate crops yearly to disrupt disease cycles.",
]

def _pick_tips(bank: List[str], k: int = 3) -> List[str]:
    r = random.Random(os.environ.get("TIPS_SEED", None))
    k = max(1, min(k, len(bank)))
    return r.sample(bank, k)

def generate_tips(crop_name: str, confidence: float) -> str:
    name = (crop_name or "").lower()
    matched_key = None
    for key in _TIP_BANK.keys():
        if key in name:
            matched_key = key
            break

    species_tips = _TIP_BANK.get(matched_key, [])
    lines = [
        f"Predicted crop: {crop_name} (confidence {confidence:.2f}).",
        "Actionable tips:",
    ]
    for tip in _pick_tips(species_tips or _GENERAL, k=2):
        lines.append(f"- {tip}")
    for tip in _pick_tips(_GENERAL, k=2):
        lines.append(f"- {tip}")
    return " ".join(lines)
