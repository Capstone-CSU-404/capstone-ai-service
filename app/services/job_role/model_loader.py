from functools import lru_cache
from pathlib import Path
import json

REPO_ROOT      = Path(__file__).resolve().parents[3]
MODEL_PATH     = REPO_ROOT / "models" / "job_role_model.keras"
ENCODER_PATH   = REPO_ROOT / "models" / "label_encoder.pkl"
TFIDF_PATH     = REPO_ROOT / "models" / "tfidf_vectorizer.pkl"
SKILL_MAP_PATH = REPO_ROOT / "models" / "skills_freq_per_role.json"

@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    try:
        import tensorflow as tf
    except Exception as exc:
        raise RuntimeError("Tensorflow is required") from exc
    return tf.keras.models.load_model(MODEL_PATH)

@lru_cache(maxsize=1)
def _load_encoder():
    if not ENCODER_PATH.exists():
        raise FileNotFoundError(f"Encoder file not found at {ENCODER_PATH}")
    try:
        import joblib
    except Exception as exc:
        raise RuntimeError("Joblib is required") from exc
    return joblib.load(ENCODER_PATH)

@lru_cache(maxsize=1)
def _load_tfidf():
    if not TFIDF_PATH.exists():
        raise FileNotFoundError(f"Encoder file not found at {TFIDF_PATH}")
    try: 
        import joblib
    except Exception as exc:
        raise RuntimeError("Joblib is required") from exc
    return joblib.load(TFIDF_PATH)
    

@lru_cache(maxsize=1)
def _load_skill_map() -> dict[str, dict[str, int]]:
    if not SKILL_MAP_PATH.exists():
        raise FileNotFoundError(f"Skills path not found at {SKILL_MAP_PATH}")
    with open(SKILL_MAP_PATH) as f:
        return json.load(f)