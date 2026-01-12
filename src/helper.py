from typing import List, Tuple
import os
import numpy as np
from joblib import load
import pandas as pd

# icl ChatGPT told me about this one, it makes the import work no matter how you run it
try:
    # Prefer absolute import when running as a script from this folder
    from audiofunctions import extract_features_from_file
except Exception:
    # Fallback to package-relative import when running as a package
    from .audiofunctions import extract_features_from_file  

# A feature vector (for future use) is defined as a numpy array of N elements eg: Fv = [f1, f2, f3,.., fN]

# IMPORTANT!! Features must be kept in this order (or the same order as your training data if you decide to use this)
FEATURE_NAMES: List[str] = [
    "pitch_mean",
    "pitch_std",
    "pitch_min",
    "pitch_max",
    "pitch_range",
    "rms_mean",
    "rms_std",
    "rms_max",
    "mfcc1_mean",
    "mfcc1_std",
    "mfcc2_mean",
    "mfcc2_std",
    "mfcc3_mean",
    "mfcc3_std",
    "mfcc4_mean",
    "mfcc4_std",
    "mfcc5_mean",
    "mfcc5_std",
    "mfcc6_mean",
    "mfcc6_std",
    "mfcc7_mean",
    "mfcc7_std",
    "mfcc8_mean",
    "mfcc8_std",
    "mfcc9_mean",
    "mfcc9_std",
    "mfcc10_mean",
    "mfcc10_std",
    "mfcc11_mean",
    "mfcc11_std",
    "mfcc12_mean",
    "mfcc12_std",
    "mfcc13_mean",
    "mfcc13_std",
    "spec_centroid_mean",
    "spec_centroid_std",
    "spec_bandwidth_mean",
    "spec_bandwidth_std",
    "zcr_mean",
    "zcr_std",
    "tempo_bpm",
    "onset_rate_per_s",
    "hnr_mean",
    "hnr_std",
] # all the column heads that need to be taken from the extracted features


def features_dict_to_vector(features: dict, feature_names: List[str] = FEATURE_NAMES) -> np.ndarray:
    # produces a np.array() from the dict of features
    row = []
    for name in feature_names:
        value = features.get(name, np.nan) # returns nan if feature not found
        row.append(value)
    return np.asarray([row], dtype=float)


def get_feature_vector(filepath: str) -> Tuple[np.ndarray, List[str]]:
    if not os.path.isfile(filepath): # check if file exists
        raise FileNotFoundError(f"File not found: {filepath}")

    features = extract_features_from_file(filepath)
    vector = features_dict_to_vector(features, FEATURE_NAMES)

    return vector, FEATURE_NAMES


def predict_with_joblib(model_path: str, filepath: str):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = load(model_path)
    vector, names = get_feature_vector(filepath)
    
    df = pd.DataFrame(vector, columns=names)
    prediction = model.predict(df)
    result = {"prediction": prediction}
    return result


