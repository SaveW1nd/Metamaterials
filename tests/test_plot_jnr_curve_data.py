from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


import sys

PLOT_DIR = ROOT / "plot"
if str(PLOT_DIR) not in sys.path:
    sys.path.insert(0, str(PLOT_DIR))

from render_top5_jnr_curve import load_curve_data


class TestPlotJnrCurveData(unittest.TestCase):
    def test_load_curve_data_reads_expected_shape(self) -> None:
        payload = load_curve_data()

        self.assertEqual(payload["top5_seeds"], [20260331, 20260339, 20260336, 20260340, 20260335])
        self.assertEqual(len(payload["per_jnr"]), 31)
        self.assertEqual(payload["per_jnr"][0]["jnr_db"], -10.0)
        self.assertEqual(payload["per_jnr"][-1]["jnr_db"], 20.0)

    def test_load_curve_data_rejects_unsorted_points(self) -> None:
        bad_payload = {
            "top5_seeds": [1, 2, 3, 4, 5],
            "per_jnr": [{"jnr_db": 1.0}, {"jnr_db": 0.0}] + [{"jnr_db": float(i)} for i in range(2, 31)],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.json"
            path.write_text(json.dumps(bad_payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_curve_data(path)
