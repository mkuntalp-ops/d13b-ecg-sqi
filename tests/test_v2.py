"""
Checks for the v2 analysis changes (revision of 2026-11-05):
- the model input matrix drops the two constant SQIs, and the tabular models reject a matrix of the wrong width;
- the zero-shot d1-3B prompt text is identical whether the SQI dictionary holds 13 or the 11 model inputs (the
  prompt shows the two constants, which equal the defaults);
- the v2 BUT QDB cohort removes exactly the 59 windows of 105001 outside the official intervals and keeps every
  other window with its v1 fold;
- the full CinC 2011 Set A file has 998 labelled records with the official label counts;
- the official BUT QDB train/test split (F1b) is subject-disjoint and covers every v2 window; threshold transfer uses
  the v2 denominators (988 / 998 / 8,512).

Run from the project root: .venv/bin/python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from features.sqi_extractor import (MODEL_INPUT_FEATURE_NAMES, SQI_FEATURE_NAMES,  # noqa: E402
                                    extract_sqi_features, format_sqi_for_prompt)
from models.tabular_baselines import SQIGradientBoosting, SQILogisticRegression  # noqa: E402

V1_MANIFEST = ROOT / "results" / "manifest_butqdb_1000_cohort.csv"
V2_MANIFEST = ROOT / "results" / "v2_2026-11" / "manifest_butqdb_1000_cohort.csv"
CINC11_FULL = ROOT / "cache" / "datasets" / "cinc2011" / "cinc2011_setA_leadII_full.csv"


class TestModelInputs(unittest.TestCase):
    def test_constants_excluded(self):
        self.assertEqual(len(MODEL_INPUT_FEATURE_NAMES), 11)
        self.assertEqual(list(MODEL_INPUT_FEATURE_NAMES), list(SQI_FEATURE_NAMES[:11]))
        self.assertNotIn("duration_sec", MODEL_INPUT_FEATURE_NAMES)
        self.assertNotIn("sampling_rate", MODEL_INPUT_FEATURE_NAMES)

    def test_tabular_models_check_width(self):
        rng = np.random.default_rng(0)
        y = np.r_[np.zeros(20), np.ones(20)]
        for cls in (SQILogisticRegression, SQIGradientBoosting):
            with self.assertRaises(ValueError):
                cls().fit(rng.normal(size=(40, 13)), y)
            m = cls().fit(rng.normal(size=(40, 11)), y)
            with self.assertRaises(ValueError):
                m.predict_proba(rng.normal(size=(5, 13)))
            self.assertEqual(m.predict_proba(rng.normal(size=(5, 11))).shape, (5,))

    def test_d1_prompt_unchanged(self):
        x = np.sin(2 * np.pi * 1.2 * np.arange(5000) / 500)
        s = extract_sqi_features(x, 500)
        full = {k: s[k] for k in SQI_FEATURE_NAMES}
        reduced = {k: s[k] for k in MODEL_INPUT_FEATURE_NAMES}
        self.assertEqual(format_sqi_for_prompt(full), format_sqi_for_prompt(reduced))


@unittest.skipUnless(V1_MANIFEST.exists() and V2_MANIFEST.exists(), "v2 cohort not set up")
class TestV2Cohort(unittest.TestCase):
    def setUp(self):
        self.v1 = pd.read_csv(V1_MANIFEST)
        self.v2 = pd.read_csv(V2_MANIFEST)

    def test_counts(self):
        self.assertEqual(len(self.v2), 988)
        self.assertEqual(int(self.v2.label.sum()), 238)
        self.assertEqual(self.v2.subject_id.nunique(), 15)
        self.assertFalse(self.v2.segment_id.duplicated().any())

    def test_only_excluded_stretch_removed(self):
        removed = self.v1[~self.v1.segment_id.isin(self.v2.segment_id)]
        self.assertEqual(len(removed), 59)
        self.assertEqual(set(removed.record_name.str[:6]), {"105001"})
        # the excluded stretch is the first 13 h of 105001 (official interval starts at sample 46,800,001)
        self.assertTrue((removed.end_sample <= 46_800_000).all())

    def test_folds_unchanged_and_subject_disjoint(self):
        m = self.v2.merge(self.v1[["segment_id", "fold_id"]], on="segment_id", suffixes=("", "_v1"))
        self.assertEqual(len(m), len(self.v2))
        self.assertTrue((m.fold_id == m.fold_id_v1).all())
        self.assertEqual(int((self.v2.groupby("subject_id").fold_id.nunique() > 1).sum()), 0)


@unittest.skipUnless(CINC11_FULL.exists(), "full CinC 2011 file not extracted")
class TestCinc2011Full(unittest.TestCase):
    def test_counts(self):
        df = pd.read_csv(CINC11_FULL)
        self.assertEqual(len(df), 998)
        self.assertEqual(df.label.value_counts().to_dict(), {0: 773, 1: 225})
        self.assertFalse(df.record_name.duplicated().any())
        self.assertTrue((df.sampling_rate == 500).all() and (df.duration_sec == 10).all())


class TestCalibration(unittest.TestCase):
    """Calibration intercept/slope against an independent fit (scipy minimisation of the log-likelihood)."""

    def setUp(self):
        sys.path.insert(0, str(ROOT / "evaluation"))
        from decision_calibration import calibration_intercept, calibration_slope, decision, logit
        self.ci, self.cs, self.dec, self.logit = calibration_intercept, calibration_slope, decision, logit
        rng = np.random.default_rng(7)
        z_true = rng.normal(0, 1.5, 4000)
        self.y = (rng.uniform(size=4000) < 1 / (1 + np.exp(-z_true))).astype(int)
        self.z_true = z_true

    def test_perfect_calibration(self):
        p = 1 / (1 + np.exp(-self.z_true))
        self.assertAlmostEqual(self.cs(self.y, p), 1.0, delta=0.1)
        self.assertAlmostEqual(self.ci(self.y, p), 0.0, delta=0.1)

    def test_known_miscalibration(self):
        # predictions twice as extreme and shifted up by 1 on the logit scale -> slope ~0.5, intercept < 0
        p = 1 / (1 + np.exp(-(2 * self.z_true + 1)))
        self.assertAlmostEqual(self.cs(self.y, p), 0.5, delta=0.05)
        self.assertLess(self.ci(self.y, p), -0.3)

    def test_against_scipy(self):
        from scipy.optimize import minimize
        p = np.clip(1 / (1 + np.exp(-(1.3 * self.z_true - 0.4))), 0, 1)
        z = self.logit(p)
        y = self.y

        def nll(b, off_only):
            eta = (b[0] + z) if off_only else (b[0] + b[1] * z)
            return np.sum(np.logaddexp(0, eta) - y * eta)
        a = minimize(lambda b: nll(b, True), [0.0], method="BFGS").x[0]
        b = minimize(lambda b: nll(b, False), [0.0, 1.0], method="BFGS").x[1]
        self.assertAlmostEqual(self.ci(y, p), a, places=4)
        self.assertAlmostEqual(self.cs(y, p), b, places=4)

    def test_decision_counts(self):
        from sklearn.metrics import confusion_matrix
        p = np.r_[0.0, 1.0, np.linspace(0, 1, 100)]
        y = (np.arange(102) % 3 == 0).astype(int)
        d = self.dec(y, p)
        tn, fp, fn, tp = confusion_matrix(y, (p >= 0.5).astype(int), labels=[0, 1]).ravel()
        self.assertEqual((d["tp"], d["fp"], d["tn"], d["fn"]), (tp, fp, tn, fn))
        self.assertEqual(d["tp"] + d["fp"] + d["tn"] + d["fn"], len(y))
        self.assertAlmostEqual(d["npv"], tn / (tn + fn))


class TestRobustFlatline(unittest.TestCase):
    def setUp(self):
        from features.sqi_extractor import robust_flatline_and_saturation
        self.f = lambda x: robust_flatline_and_saturation(x, 500)[0]
        t = np.arange(5000) / 500
        self.ecg = 0.05 * np.sin(2 * np.pi * 0.3 * t)
        for k in range(10):
            self.ecg[k * 500 + 250:k * 500 + 260] += 1.0

    def test_quantization_is_not_flat(self):
        q = np.round(self.ecg / 0.005) * 0.005
        self.assertEqual(self.f(self.ecg), 0.0)
        self.assertEqual(self.f(q), 0.0)

    def test_lead_off_is_flat(self):
        x = self.ecg.copy()
        x[1000:3500] = 0.0
        self.assertAlmostEqual(self.f(x), 0.5, places=3)
        self.assertEqual(self.f(np.zeros(5000)), 1.0)



V2 = ROOT / "results" / "v2_2026-11"


@unittest.skipUnless((V2 / "butqdb_official_split_assignment.csv").exists(), "official split not written")
class TestOfficialSplit(unittest.TestCase):
    """F1b (handoff item 1, 2026-11-07): official records, subject-disjoint, every v2 window assigned once."""

    def test_assignment(self):
        a = pd.read_csv(V2 / "butqdb_official_split_assignment.csv")
        man = pd.read_csv(V2_MANIFEST)
        self.assertEqual(list(a.segment_id), list(man.segment_id))
        self.assertEqual(set(a.split), {"train", "test"})  # no v2 window lies outside the official intervals
        self.assertFalse(set(a.subject_id[a.split == "train"]) & set(a.subject_id[a.split == "test"]))
        test_rec = set(man.record_name[a.split.values == "test"].str[:6].astype(int))
        self.assertEqual(test_rec, {105001, 118001, 121001, 122001, 123001, 124001, 125001})

    @unittest.skipUnless((V2 / "tabpfn_official_split_predictions.csv").exists(), "TabPFN F1b not run")
    def test_tabpfn_on_test_windows_only(self):
        a = pd.read_csv(V2 / "butqdb_official_split_assignment.csv")
        t = pd.read_csv(V2 / "tabpfn_official_split_predictions.csv")
        self.assertEqual(set(t.segment_id), set(a.segment_id[a.split == "test"]))
        self.assertTrue(t.prob_unusable.between(0, 1).all())


@unittest.skipUnless((V2 / "threshold_transfer_summary.csv").exists(), "threshold transfer not run")
class TestThresholdTransferUnits(unittest.TestCase):
    def test_denominators(self):
        t = pd.read_csv(V2 / "threshold_transfer_summary.csv")
        n = (t.n_pos + t.n_neg).groupby(t.dataset).unique()
        self.assertEqual(list(n["CinC2011 all 998 records (lead II)"]), [998])
        self.assertEqual(list(n["CinC2017 all (record max)"]), [8512])
        self.assertEqual(list(n["BUT_QDB pooled OOF (selection set)"]), [988])


class TestBootstrapPValueFloor(unittest.TestCase):
    """Code audit 2026-11-11, CODE-003: a resampling p-value is never reported below its resolution."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "evaluation"))
        from paired_comparisons import cluster_bootstrap_paired
        cls.fn = staticmethod(cluster_bootstrap_paired)
        rng = np.random.default_rng(0)
        cls.cluster = np.repeat(np.arange(12), 40)
        cls.y = np.tile(np.r_[np.zeros(30), np.ones(10)], 12).astype(int)
        cls.good = cls.y + rng.normal(0, 0.1, len(cls.y))     # separates the classes in every cluster
        cls.noise = rng.normal(0, 1, len(cls.y))
        cls.fair = cls.y + rng.normal(0, 1.5, len(cls.y))
        cls.fair2 = cls.y + rng.normal(0, 1.5, len(cls.y))

    def test_floor_when_no_resample_crosses_zero(self):
        d, lo, hi, p = self.fn(self.y, self.good, self.noise, self.cluster, B=500)
        self.assertGreater(lo, 0)
        self.assertAlmostEqual(p, 1 / 500)

    def test_other_p_values_unchanged(self):
        from sklearn.metrics import roc_auc_score
        _, _, _, p = self.fn(self.y, self.fair, self.fair2, self.cluster, B=500, seed=42)
        rng, ug = np.random.default_rng(42), np.unique(self.cluster)
        idx = {c: np.where(self.cluster == c)[0] for c in ug}
        d = []
        for _ in range(500):
            i = np.concatenate([idx[c] for c in rng.choice(ug, len(ug))])
            d.append(roc_auc_score(self.y[i], self.fair[i]) - roc_auc_score(self.y[i], self.fair2[i]))
        d = np.array(d)
        expected = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
        self.assertGreater(expected, 1 / 500)
        self.assertAlmostEqual(p, expected)


class TestAuditSensitivity(unittest.TestCase):
    """evaluation/audit_sensitivity.py on synthetic inputs (code audit 2026-11-11)."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "evaluation"))
        import audit_sensitivity
        cls.a = audit_sensitivity

    def test_constant_lead_records(self):
        src = pd.DataFrame({"record_name": [1, 2, 3], "signal_std": [0.0, 0.2, 0.0]})
        self.assertEqual(self.a.constant_lead_records(src), {"1", "3"})

    def test_paired_family(self):
        rng = np.random.default_rng(1)
        y = np.r_[np.zeros(200), np.ones(100)].astype(int)
        s = pd.DataFrame({"unit": np.arange(300), "y": y, "strong": y + rng.normal(0, 0.5, 300),
                          "medium": y + rng.normal(0, 1.5, 300), "chance": rng.normal(0, 1, 300)})
        out = self.a.paired_family(s)
        self.assertEqual(len(out), 3)
        self.assertEqual(set(out.best_model), {"strong"})
        self.assertTrue((out.p_holm >= out.p - 1e-15).all())
        vs_best = out[(out.model_a == "strong") | (out.model_b == "strong")]
        self.assertEqual(len(vs_best), 2)
        self.assertTrue(vs_best.p_holm_vs_best.notna().all())
        self.assertTrue(out[(out.model_a != "strong") & (out.model_b != "strong")].p_holm_vs_best.isna().all())
        row = out[(out.model_a == "strong") & (out.model_b == "chance")].iloc[0]
        self.assertAlmostEqual(row.delta, row.auroc_a - row.auroc_b)
        self.assertLess(row.p_holm, 0.05)

    def test_per_subject_auroc(self):
        s = pd.DataFrame({"unit": range(9), "cluster": ["a"] * 4 + ["b"] * 3 + ["c"] * 2,
                          "y": [0, 0, 1, 1, 0, 0, 0, 0, 1], "m": [0.1, 0.2, 0.8, 0.9, 0.3, 0.4, 0.5, 0.9, 0.1]})
        out = self.a.per_subject_auroc(s).set_index("subject")
        self.assertEqual(set(out.index), {"a", "c", "pooled_mixed_subjects", "all_subjects"})  # b has one class
        self.assertEqual(out.loc["a", "auroc"], 1.0)
        self.assertEqual(out.loc["c", "auroc"], 0.0)
        self.assertEqual(int(out.loc["pooled_mixed_subjects", "n_windows"]), 6)
        self.assertEqual(int(out.loc["all_subjects", "n_unusable"]), 3)

    def test_length_stratified(self):
        from sklearn.metrics import roc_auc_score
        rng = np.random.default_rng(2)
        rows = []
        for r in range(120):
            lab, n = int(r % 4 == 0), 1 + r % 3
            rows += [{"record_name": f"r{r}", "label": lab, "unit_idx": k, "m": lab + rng.normal(0, 1)} for k in range(n)]
        w = pd.DataFrame(rows)
        out = self.a.length_stratified(w, ["m"])
        n = out.drop_duplicates("stratum").set_index("stratum").n_recordings
        self.assertEqual(int(n["all"]), 120)
        self.assertEqual(int(n["1_window"] + n["2_windows"] + n["3_windows"]), 120)
        g = w.groupby("record_name")
        expect = roc_auc_score(g.label.first(), g.m.max())
        got = out[(out.stratum == "all") & (out.aggregation == "max")].auroc.iloc[0]
        self.assertAlmostEqual(got, expect)
        one = out[out.stratum == "1_window"].set_index("aggregation").auroc   # one window: every rule is the same
        self.assertAlmostEqual(one["max"], one["mean"])
        self.assertAlmostEqual(one["max"], one["first_window"])

    def test_rank_tables(self):
        cis = pd.DataFrame({"family": ["A"] * 3 + ["B"] * 3, "model": ["x", "y", "z"] * 2, "auroc": [0.9, 0.8, 0.7, 0.6, 0.7, 0.8]})
        ranks, rho = self.a.rank_tables(cis)
        r = ranks.set_index("model")
        self.assertEqual(list(r.loc[["x", "y", "z"], "rank | A"]), [1.0, 2.0, 3.0])
        self.assertEqual(list(r.loc[["x", "y", "z"], "rank | B"]), [3.0, 2.0, 1.0])
        self.assertAlmostEqual(rho.spearman_rho.iloc[0], -1.0)
        with self.assertRaises(ValueError):
            self.a.rank_tables(cis.iloc[:-1])

    def test_patient_check(self):
        sample = pd.DataFrame({"ecg_id": [1, 2, 3], "stratum": ["clean", "clean", "global"]})
        meta = pd.DataFrame({"ecg_id": [1, 2, 3, 4], "patient_id": [10, 10, 11, 12]})
        out = self.a.patient_check(sample, meta).set_index("stratum")
        self.assertEqual(int(out.loc["clean", "patients_with_more_than_one_record"]), 1)
        self.assertEqual(int(out.loc["all", "n_patients"]), 2)
        with self.assertRaises(ValueError):
            self.a.patient_check(pd.DataFrame({"ecg_id": [9], "stratum": ["clean"]}), meta)


class TestHarmonisedPipeline(unittest.TestCase):
    """Safeguards of the harmonised-pipeline analysis (PAPER-002; pre-stated 2026-11-11)."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "experiments"))
        import harmonised_inputs
        import harmonised_pipeline
        cls.h, cls.hp = harmonised_inputs, harmonised_pipeline

    def test_feature_lists(self):
        self.assertEqual(self.h.ORIGINAL, MODEL_INPUT_FEATURE_NAMES)
        self.assertEqual(len(self.h.HARMONISED), 11)
        self.assertNotIn("flatline_fraction", self.h.HARMONISED)
        self.assertEqual(self.h.HARMONISED.index("flatline_fraction_robust"), self.h.ORIGINAL.index("flatline_fraction"))
        self.assertEqual(self.h.CLIP, 3.0)

    def test_zclip_uses_training_statistics_and_bound(self):
        rng = np.random.default_rng(0)
        train = np.column_stack([rng.normal(5, 2, 500), np.full(500, 0.25), rng.normal(0, 1e-4, 500)])
        mean, sd = self.h.zclip_fit(train)
        self.assertEqual(sd[1], 1.0)                       # constant training column: SD replaced by 1
        test = np.array([[5.0, 0.25, 0.0], [1e6, 0.25, 1.0], [-1e6, 9.0, np.nan]])
        z = self.h.zclip_apply(test, mean, sd)
        self.assertTrue((np.abs(z) <= 3.0).all())
        self.assertAlmostEqual(z[0, 0], (5.0 - mean[0]) / sd[0])
        self.assertEqual((z[1, 0], z[2, 0]), (3.0, -3.0))   # a value thousands of SDs away is bounded
        self.assertEqual(z[1, 2], 3.0)
        self.assertEqual(z[0, 1], 0.0)
        self.assertTrue(np.isfinite(z).all())

    def test_constant_rule(self):
        p = self.hp.apply_constant_rule(np.array([0.2, 0.0, 0.9]), np.array([False, True, False]))
        self.assertEqual(list(p), [0.2, 1.0, 0.9])


@unittest.skipUnless((V2 / "model_cis.csv").exists() and (V2 / "external_window_predictions.csv").exists(),
                     "v2 outputs not present")
class TestCinc2017WindowTable(unittest.TestCase):
    """paired_comparisons.cinc2017_windows() (split out of families() on 2026-11-11) gives the stored AUROCs."""

    def test_recording_maximum_matches_stored_auroc(self):
        import os
        import subprocess
        code = ("import sys; sys.path.insert(0, 'evaluation'); import paired_comparisons as pc; "
                "from sklearn.metrics import roc_auc_score; w, cols = pc.cinc2017_windows(); "
                "r = w.groupby('record_name').agg({**{m: 'max' for m in cols}, 'label': 'first'}); "
                "print(len(r), len(cols), ';'.join(f'{m}={roc_auc_score(r.label, r[m]):.12f}' for m in cols))")
        env = {**os.environ, "ECG_RESULTS_DIR": str(V2), "PYTHONDONTWRITEBYTECODE": "1"}
        res = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
        n_rec, n_models, pairs = res.stdout.strip().split(" ", 2)
        self.assertEqual((int(n_rec), int(n_models)), (8512, 11))
        cis = pd.read_csv(V2 / "model_cis.csv")
        stored = cis[cis.family == "CinC2017 (8,512 recordings)"].set_index("model").auroc
        for pair in pairs.split(";"):
            m, v = pair.split("=")
            self.assertAlmostEqual(float(v), stored[m], places=10)


class TestTableBlocks(unittest.TestCase):
    """Multi-part generated tables must reach the manuscript whole (review of 2026-11-12)."""

    def test_parts_separated_by_blank_lines_are_kept(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))
        from sync_manuscript_tables import table_blocks
        lines = ["## Table 1", "", "| a |", "| :- |", "| 1 |", "", "| b |", "| :- |", "| 2 |", "", "A note.", "",
                 "## Table 2", "", "| c |", "| :- |", "| 3 |", "## Table 3", "| d |", "| :- |", "| 4 |", "", ""]
        blocks = table_blocks(lines)
        self.assertEqual(blocks["Table 1"], ["| a |", "| :- |", "| 1 |", "", "| b |", "| :- |", "| 2 |"])
        self.assertEqual(blocks["Table 2"], ["| c |", "| :- |", "| 3 |"])
        self.assertEqual(blocks["Table 3"], ["| d |", "| :- |", "| 4 |"])


if __name__ == "__main__":
    unittest.main()
