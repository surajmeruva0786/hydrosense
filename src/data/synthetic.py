"""Synthetic hydrophone-like audio generator.

The real ShipsEar corpus is gated behind a request-access form at the
University of Vigo (README §6) and is not redistributable, so it cannot be
vendored into this repository. Every part of the HydroSense pipeline —
preprocessing, training, TimeGAN, XAI — is nonetheless runnable end-to-end
against data generated here: physically-motivated but synthetic underwater
noise with per-category harmonic structure loosely modelled on the five
ShipsEar operational categories (README §6, class taxonomy).

This is a development/CI fixture, **not** a substitute for real acoustic
data. Numbers produced from synthetic data are not meaningful benchmarks;
see README §16 and `data/README.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Rough per-category spectral profile: (fundamental_hz, n_harmonics, broadband_gain)
# A = small diesel fishing boats -> low fundamental, strong low harmonics
# B = motor/pilot/sailboats -> mid fundamental, fewer harmonics
# C = passenger ferries -> broadband, moderate harmonics
# D = ocean liners / ro-ro -> very low fundamental (large slow shaft), heavy broadband
# E = ambient / no vessel -> harmonics ~ 0, pure coloured noise
CLASS_PROFILES: dict[str, dict] = {
    "A": {"fundamental_hz": 12.0, "n_harmonics": 8, "harmonic_gain": 0.8, "broadband_gain": 0.15},
    "B": {"fundamental_hz": 25.0, "n_harmonics": 5, "harmonic_gain": 0.5, "broadband_gain": 0.20},
    "C": {"fundamental_hz": 8.0, "n_harmonics": 12, "harmonic_gain": 0.4, "broadband_gain": 0.35},
    "D": {"fundamental_hz": 4.0, "n_harmonics": 15, "harmonic_gain": 0.9, "broadband_gain": 0.45},
    "E": {"fundamental_hz": 0.0, "n_harmonics": 0, "harmonic_gain": 0.0, "broadband_gain": 0.10},
}

CLASS_NAMES = list(CLASS_PROFILES.keys())


@dataclass
class SyntheticConfig:
    sample_rate: int = 16000
    seed: int = 42


def _pink_noise(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Cheap approximation of pink (1/f) noise via FFT-domain shaping."""
    white = rng.standard_normal(n_samples)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
    spectrum = spectrum / np.sqrt(freqs)
    pink = np.fft.irfft(spectrum, n=n_samples)
    return pink / (np.max(np.abs(pink)) + 1e-8)


def generate_synthetic_recording(
    class_name: str,
    duration_s: float,
    sample_rate: int = 16000,
    seed: int = 42,
    amplitude_modulation_hz: float = 1.5,
) -> np.ndarray:
    """Synthesize one mono waveform for `class_name` (a key of `CLASS_PROFILES`).

    Combines a harmonic series (propeller blade-rate + shaft harmonics),
    slow amplitude modulation (blade passage), and coloured broadband noise
    (flow/machinery noise), normalised to [-1, 1].
    """
    if class_name not in CLASS_PROFILES:
        raise ValueError(f"Unknown class '{class_name}'. Expected one of {CLASS_NAMES}.")

    rng = np.random.default_rng(seed)
    profile = CLASS_PROFILES[class_name]
    n_samples = int(duration_s * sample_rate)
    t = np.arange(n_samples) / sample_rate

    signal = np.zeros(n_samples, dtype=np.float64)

    if profile["n_harmonics"] > 0:
        for h in range(1, profile["n_harmonics"] + 1):
            freq = profile["fundamental_hz"] * h
            if freq >= sample_rate / 2:
                break
            phase = rng.uniform(0, 2 * np.pi)
            amp = profile["harmonic_gain"] / h
            signal += amp * np.sin(2 * np.pi * freq * t + phase)

        modulation = 1.0 + 0.3 * np.sin(2 * np.pi * amplitude_modulation_hz * t)
        signal *= modulation

    noise = _pink_noise(n_samples, rng) * profile["broadband_gain"]
    signal = signal + noise

    peak = np.max(np.abs(signal)) + 1e-8
    signal = signal / peak * 0.9
    return signal.astype(np.float32)


def generate_synthetic_dataset(
    output_dir,
    recordings_per_class: dict[str, int] | None = None,
    duration_range_s: tuple[float, float] = (20.0, 60.0),
    sample_rate: int = 16000,
    seed: int = 42,
):
    """Write a small ShipsEar-shaped synthetic corpus to `output_dir/*.wav`.

    Returns a list of metadata dicts (one per recording) suitable for
    writing straight to a manifest CSV. Import-time dependency on
    `soundfile` is kept local to this function so the rest of `src.data`
    stays importable without an audio backend installed.
    """
    import soundfile as sf

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if recordings_per_class is None:
        # Mirrors the real ShipsEar class imbalance (README §6) at small scale.
        recordings_per_class = {"A": 6, "B": 8, "C": 10, "D": 5, "E": 4}

    rng = np.random.default_rng(seed)
    records = []
    rec_idx = 0
    for class_name, n_recordings in recordings_per_class.items():
        for i in range(n_recordings):
            duration = float(rng.uniform(*duration_range_s))
            rec_seed = seed + rec_idx
            waveform = generate_synthetic_recording(
                class_name, duration, sample_rate=sample_rate, seed=rec_seed
            )
            filename = f"{class_name}_{i:03d}.wav"
            filepath = output_dir / filename
            sf.write(filepath, waveform, sample_rate)

            records.append(
                {
                    "recording_id": filename.replace(".wav", ""),
                    "file_path": str(filepath),
                    "class_name": class_name,
                    "duration_s": duration,
                }
            )
            rec_idx += 1

    return records
