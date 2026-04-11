from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baseline_wei2018 import estimate_period_from_sequence


def test_estimate_period_from_sequence_recovers_known_period() -> None:
    sequence = np.tile(np.array([0.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32), 20)

    period = estimate_period_from_sequence(sequence, min_lag=2, max_lag=12)

    assert period == 5
