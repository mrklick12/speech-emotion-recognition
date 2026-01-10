import librosa
import numpy as np
import parselmouth

# NOTE: conventional variables for librosa y, sr have been replaced with audio, sampling_rate for clarity

""""Audiofunctions.py's job is to return the audio features from a .wav file as a dictionary
The details of what is extracted are in in README.md, please refer to that for full project documentation
"""
def load_audio(filepath: str):
    # sr=None preserves original sampling rate
    audio, sampling_rate = librosa.load(filepath, sr=None, mono=True)
    return audio, sampling_rate

def pitch_features(audio: np.ndarray, sampling_rate: int) -> dict:
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
    )
    # f0 contains NaN for unvoiced frames
    f0_valid = f0[np.isnan(f0) == False]
    if len(f0_valid) == 0:
        return {"pitch_mean": np.nan, "pitch_std": np.nan, "pitch_min": np.nan, "pitch_max": np.nan, "pitch_range": np.nan}

    return {
        "pitch_mean": float(np.mean(f0_valid)),
        "pitch_std": float(np.std(f0_valid)),
        "pitch_min": float(np.min(f0_valid)),
        "pitch_max": float(np.max(f0_valid)),
        "pitch_range": float(np.max(f0_valid) - np.min(f0_valid)),
    }

def energy_features(audio: np.ndarray, sampling_rate: int) -> dict:
    rms = librosa.feature.rms(y=audio)[0]  # shape: (frames,)
    return {
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "rms_max": float(np.max(rms)),
    }

def mfcc_features(audio: np.ndarray, sampling_rate: int, n_mfcc: int = 13) -> dict:
    mfcc = librosa.feature.mfcc(y=audio, sr=sampling_rate, n_mfcc=n_mfcc)  # (n_mfcc, frames)
    feats = {}
    for i in range(n_mfcc):
        feats[f"mfcc{i+1}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc{i+1}_std"] = float(np.std(mfcc[i]))
    return feats

def spectral_centroid_features(audio: np.ndarray, sampling_rate: int) -> dict:
    sc = librosa.feature.spectral_centroid(y=audio, sr=sampling_rate)[0]
    return {
        "spec_centroid_mean": float(np.mean(sc)),
        "spec_centroid_std": float(np.std(sc)),
    }

def spectral_bandwidth_features(audio: np.ndarray, sampling_rate: int) -> dict:
    sbw = librosa.feature.spectral_bandwidth(y=audio, sr=sampling_rate)[0]
    return {
        "spec_bandwidth_mean": float(np.mean(sbw)),
        "spec_bandwidth_std": float(np.std(sbw)),
    }

def zcr_features(audio: np.ndarray) -> dict:
    zcr = librosa.feature.zero_crossing_rate(y=audio)[0]
    return {
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr)),
    }

def tempo_speechrate_features(audio: np.ndarray, sampling_rate: int) -> dict:
    # Onset-based tempo estimate (in BPM)
    tempo_bpm = librosa.feature.tempo(y=audio, sr=sampling_rate)
    tempo_bpm = float(tempo_bpm[0]) if len(tempo_bpm) else np.nan

    # Onset rate: onsets per second (more interpretable for speech)
    onsets = librosa.onset.onset_detect(y=audio, sr=sampling_rate, units="time")
    duration = librosa.get_duration(y=audio, sr=sampling_rate)
    onset_rate = float(len(onsets) / duration) if duration > 0 else np.nan

    return {
        "tempo_bpm": tempo_bpm,
        "onset_rate_per_s": onset_rate,
    }

def hnr_features(filepath: str) -> dict:
    try:
        snd = parselmouth.Sound(filepath)
        harmonicity = snd.to_harmonicity_cc()
        hnr_values = harmonicity.values[0]  # one row
        # Remove -inf and nan
        hnr_values = hnr_values[np.isfinite(hnr_values)]
        if len(hnr_values) == 0:
            return {"hnr_mean": np.nan, "hnr_std": np.nan}
        return {
            "hnr_mean": float(np.mean(hnr_values)),
            "hnr_std": float(np.std(hnr_values)),
        }
    except Exception:
        return {"hnr_mean": np.nan, "hnr_std": np.nan}

def extract_features_from_file(filepath: str) -> dict:
    audio, sampling_rate = load_audio(filepath)

    features = {}
    features.update(pitch_features(audio, sampling_rate))
    features.update(energy_features(audio, sampling_rate))
    features.update(mfcc_features(audio, sampling_rate, n_mfcc=13))
    features.update(spectral_centroid_features(audio, sampling_rate))
    features.update(spectral_bandwidth_features(audio, sampling_rate))
    features.update(zcr_features(audio))
    features.update(tempo_speechrate_features(audio, sampling_rate))
    features.update(hnr_features(filepath))  # uses filepath, not audio/sampling_rate

    return features # this returns a DICTIONARY of features eg: {pitch_mean: 384.4, pitch_std: 45.2, ...}