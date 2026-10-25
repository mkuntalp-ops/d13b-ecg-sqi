"""
Core checks for the ECG usability benchmark (standard-library unittest; run from the project root):

    .venv/bin/python -m unittest discover -s tests -v

They cover the metric code against scikit-learn, the paired-comparison statistics, the subject-independent
folds, unit handling, the renderer regression found on 2026-10-13, and basic SQI properties.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

from data.ecg_rendering import render_ecg_image  # noqa: E402
from data.units import to_millivolts  # noqa: E402
from evaluation.metrics import compute_usability_metrics  # noqa: E402
from features.qrs_detectors import calculate_bsqi  # noqa: E402
from features.sqi_extractor import extract_sqi_features  # noqa: E402
from paired_comparisons import delong_paired, holm  # noqa: E402

MANIFEST = ROOT / "results" / "manifest_butqdb_1000_cohort.csv"


class TestMetrics(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.y = rng.integers(0, 2, 300)
        self.p = np.clip(self.y * 0.4 + rng.uniform(0, 0.6, 300), 0, 1)

    def test_against_sklearn(self):
        m = compute_usability_metrics(self.y, self.p, threshold=0.5)
        tn, fp, fn, tp = confusion_matrix(self.y, (self.p >= 0.5).astype(int), labels=[0, 1]).ravel()
        self.assertEqual(m["tp"] + m["fp"] + m["tn"] + m["fn"], len(self.y))
        self.assertEqual((m["tp"], m["fp"], m["tn"], m["fn"]), (tp, fp, tn, fn))
        self.assertAlmostEqual(m["unusable_sensitivity"], tp / (tp + fn))
        self.assertAlmostEqual(m["usable_false_reject_rate"], fp / (fp + tn))
        self.assertAlmostEqual(m["auroc"], roc_auc_score(self.y, self.p))

    def test_single_class_auroc_is_nan(self):
        self.assertTrue(np.isnan(compute_usability_metrics(np.zeros(10), np.linspace(0, 1, 10))["auroc"]))


class TestPairedStatistics(unittest.TestCase):
    def test_delong_delta_and_identity(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 400)
        a, b = y + rng.normal(0, 1, 400), y + rng.normal(0, 1.5, 400)
        d, lo, hi, p = delong_paired(y, a, b)
        self.assertAlmostEqual(d, roc_auc_score(y, a) - roc_auc_score(y, b))
        self.assertLess(lo, d)
        self.assertGreater(hi, d)
        self.assertEqual(delong_paired(y, a, a)[3], 1.0)

    def test_holm(self):
        np.testing.assert_allclose(holm(np.array([0.01, 0.04, 0.03, 0.2])), [0.04, 0.09, 0.09, 0.2])


@unittest.skipUnless(MANIFEST.exists(), "cohort manifest not generated")
class TestCohortSplits(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_csv(MANIFEST)

    def test_subject_disjoint_folds(self):
        self.assertEqual(int((self.df.groupby("subject_id").fold_id.nunique() > 1).sum()), 0)

    def test_labels_and_counts(self):
        self.assertTrue(((self.df.original_quality_class == 3).astype(int) == self.df.label).all())
        self.assertEqual(len(self.df), 1047)
        self.assertEqual(int(self.df.label.sum()), 297)
        self.assertFalse(self.df.segment_id.duplicated().any())

    def test_no_overlapping_windows(self):
        for _, g in self.df.sort_values(["record_name", "start_sample"]).groupby("record_name"):
            self.assertTrue((g.start_sample.values[1:] >= g.end_sample.values[:-1]).all())


class TestUnitsAndRendering(unittest.TestCase):
    def test_to_millivolts(self):
        x = np.array([1000.0, -500.0])
        np.testing.assert_allclose(to_millivolts(x, "uV"), [1.0, -0.5])
        np.testing.assert_allclose(to_millivolts(x, "mV"), x)
        with self.assertRaises(ValueError):
            to_millivolts(x, "V?")

    def test_large_mv_artefact_is_not_flattened(self):
        # regression: the earlier renderer divided any signal exceeding 20 by 1,000 and drew it as a flat line
        t = np.arange(5000) / 500
        x = (np.sin(2 * np.pi * 1.2 * t) + (np.arange(5000) == 2500) * 80.0).astype(np.float32)
        img = np.asarray(render_ecg_image(x, 500).convert("L"))
        ink_rows = np.where((img < 60).any(axis=1))[0]
        self.assertGreater(ink_rows.max() - ink_rows.min(), 0.3 * img.shape[0])


class TestSQI(unittest.TestCase):
    def test_saturation_floor_and_quantized_flatline(self):
        rng = np.random.default_rng(3)
        x = rng.normal(0, 1, 5000)
        s = extract_sqi_features(x, 500)
        self.assertAlmostEqual(s["saturation_fraction"], 2 / 5000)
        self.assertLess(s["flatline_fraction"], 1e-3)
        # a smooth 1 Hz, 0.5 mV signal: continuous values give no repeats, 5 uV quantization (CinC 2011 lead II) gives many
        smooth = 0.5 * np.sin(2 * np.pi * np.arange(5000) / 500)
        self.assertLess(extract_sqi_features(smooth, 500)["flatline_fraction"], 1e-3)
        self.assertGreater(extract_sqi_features(np.round(smooth / 0.005) * 0.005, 500)["flatline_fraction"], 0.1)

    def test_bsqi_identical_detectors(self):
        peaks = np.arange(100, 5000, 400)
        self.assertAlmostEqual(calculate_bsqi(peaks, peaks, 500, tolerance_ms=50.0), 1.0)


if __name__ == "__main__":
    unittest.main()
