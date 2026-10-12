"""
Amplitude-unit harmonization for the SQI pipeline.

All SQI features are computed on millivolt signals. The sources differ:
- BUT QDB (PhysioNet 'butqdb'): header units 'uV' (microvolts).
- CinC 2011 Set A and NSTDB: header units 'mV'.

Without this conversion, `signal_std` differs by ~1000x between datasets and
cannot be compared across them (see manuscript Limitation 9).
"""

import numpy as np

_MICROVOLT_NAMES = {"uv", "µv", "microvolt", "microvolts"}
_MILLIVOLT_NAMES = {"mv", "millivolt", "millivolts"}


def to_millivolts(ecg: np.ndarray, unit: str) -> np.ndarray:
    """Return the signal in millivolts given its header unit string."""
    key = (unit or "").strip().lower()
    if key in _MILLIVOLT_NAMES:
        return np.asarray(ecg, dtype=np.float64)
    if key in _MICROVOLT_NAMES:
        return np.asarray(ecg, dtype=np.float64) / 1000.0
    raise ValueError(f"Unknown amplitude unit '{unit}'; refusing to guess the scale.")
