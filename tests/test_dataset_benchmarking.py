"""
tests/test_dataset_benchmarking.py

Unit and integration tests for Phase 2 dataset assurance enhancements:
1. Hungarian assignment and IoU bounding box matching in check_label_flipping.
2. Annotation permutation invariance (reordered boxes/labels).
3. Detailed discrepancy taxonomy (CLASS_CHANGE, BBOX_MODIFIED, MISSING_ANNOTATION, ADDITIONAL_ANNOTATION).
4. dHash perceptual hashing and near-duplicate detection.
5. Reference-calibrated data poisoning detection with robust MAD.
6. Reference-calibrated OOD detection with Mahalanobis distance.
7. Benchmark fixtures integrity and ground-truth separation.
"""

import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from src.dataset.security_checks import (
    compute_bbox_iou,
    _extract_bbox,
    compute_dhash,
    hamming_distance,
    check_label_flipping,
    check_near_duplicates,
    check_duplicate_flooding,
    check_trigger_patterns,
    check_data_poisoning,
    check_ood_samples,
)

ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = ROOT / "data" / "benchmark"


# ---------------------------------------------------------------------------
# 1. Bounding Box IoU & Extraction Unit Tests
# ---------------------------------------------------------------------------

class TestBboxAndIoU:
    def test_extract_bbox_formats(self):
        # Dict bbox [x, y, w, h] format -> converts to (x1, y1, x2, y2)
        ann_dict = {"bbox": [10.0, 20.0, 30.0, 40.0], "class_id": 0}
        assert _extract_bbox(ann_dict) == (10.0, 20.0, 40.0, 60.0)

        # Explicit center keys format -> converts to (x1, y1, x2, y2)
        ann_keys = {"x_center": 0.4, "y_center": 0.4, "width": 0.2, "height": 0.2}
        extracted = _extract_bbox(ann_keys)
        assert pytest.approx(extracted[0]) == 0.3
        assert pytest.approx(extracted[1]) == 0.3
        assert pytest.approx(extracted[2]) == 0.5
        assert pytest.approx(extracted[3]) == 0.5

        # Explicit box [x1, y1, x2, y2]
        ann_box = {"box": [0.1, 0.2, 0.3, 0.4]}
        assert _extract_bbox(ann_box) == (0.1, 0.2, 0.3, 0.4)

        # Invalid / missing format
        assert _extract_bbox({"category": "dog"}) is None

    def test_compute_bbox_iou_identical(self):
        box = (0.2, 0.2, 0.6, 0.6)
        iou = compute_bbox_iou(box, box)
        assert pytest.approx(iou, rel=1e-5) == 1.0

    def test_compute_bbox_iou_disjoint(self):
        box1 = (0.1, 0.1, 0.2, 0.2)
        box2 = (0.8, 0.8, 0.9, 0.9)
        iou = compute_bbox_iou(box1, box2)
        assert iou == 0.0

    def test_compute_bbox_iou_partial_overlap(self):
        # box1 [0, 0, 2, 2] area = 4
        # box2 [1, 0, 3, 2] area = 4
        # intersection [1, 0, 2, 2] area = 2
        # union = 4 + 4 - 2 = 6
        # IoU = 2 / 6 = 1/3 ~ 0.33333
        box1 = (0.0, 0.0, 2.0, 2.0)
        box2 = (1.0, 0.0, 3.0, 2.0)
        iou = compute_bbox_iou(box1, box2)
        assert pytest.approx(iou, rel=1e-4) == 1.0 / 3.0

    def test_compute_bbox_iou_zero_area(self):
        box1 = (0.5, 0.5, 0.5, 0.6)  # width = 0
        box2 = (0.5, 0.5, 0.7, 0.7)
        assert compute_bbox_iou(box1, box2) == 0.0


# ---------------------------------------------------------------------------
# 2. Hungarian Assignment & Label Flipping Taxonomy
# ---------------------------------------------------------------------------

class TestHungarianLabelMatching:
    def test_permutation_invariance_clean(self):
        """Annotations in different order must yield PASS and zero flips."""
        ref = {
            "img1.jpg": [
                {"bbox": [0.1, 0.1, 0.1, 0.1], "class_id": 0},
                {"bbox": [0.6, 0.6, 0.2, 0.2], "class_id": 1},
                {"bbox": [0.4, 0.4, 0.15, 0.15], "class_id": 2},
            ]
        }
        cand = {
            "img1.jpg": [
                {"bbox": [0.4, 0.4, 0.15, 0.15], "class_id": 2},
                {"bbox": [0.1, 0.1, 0.1, 0.1], "class_id": 0},
                {"bbox": [0.6, 0.6, 0.2, 0.2], "class_id": 1},
            ]
        }
        res = check_label_flipping(cand, ref)
        assert res["result"] == "PASS"
        assert res["evidence"]["flip_count"] == 0
        assert len(res["evidence"]["examples"]) == 0

    def test_class_change_detected(self):
        """Same bounding box, different class must be detected as CLASS_CHANGE."""
        ref = {
            "img1.jpg": [{"bbox": [0.4, 0.4, 0.2, 0.2], "class_id": 0}]
        }
        cand = {
            "img1.jpg": [{"bbox": [0.4, 0.4, 0.2, 0.2], "class_id": 1}]
        }
        res = check_label_flipping(cand, ref)
        assert res["result"] == "ANOMALY_DETECTED"
        assert res["evidence"]["flip_count"] == 1
        assert res["evidence"]["object_level_breakdown"]["class_changes"] == 1
        disc = res["evidence"]["examples"][0]
        assert disc["issue"] == "CLASS_CHANGE"
        assert disc["reference_class"] == 0
        assert disc["candidate_class"] == 1
        assert pytest.approx(disc["iou"], rel=1e-4) == 1.0

    def test_bbox_modified_detected(self):
        """Slight bounding box modification (IoU < 0.95, >= 0.5) must be BBOX_MODIFIED."""
        ref = {
            "img1.jpg": [{"box": [0.2, 0.2, 0.6, 0.6], "class_id": 0}]
        }
        cand = {
            "img1.jpg": [{"box": [0.25, 0.2, 0.65, 0.6], "class_id": 0}]
        }
        res = check_label_flipping(cand, ref)
        assert res["result"] == "ANOMALY_DETECTED"
        assert res["evidence"]["flip_count"] == 1
        assert res["evidence"]["object_level_breakdown"]["bbox_modifications"] == 1
        disc = res["evidence"]["examples"][0]
        assert disc["issue"] == "BBOX_MODIFIED"
        assert 0.5 <= disc["iou"] < 0.95

    def test_missing_annotation_detected(self):
        """Candidate missing one of the reference bounding boxes."""
        ref = {
            "img1.jpg": [
                {"bbox": [0.1, 0.1, 0.1, 0.1], "class_id": 0},
                {"bbox": [0.7, 0.7, 0.1, 0.1], "class_id": 1},
            ]
        }
        cand = {
            "img1.jpg": [
                {"bbox": [0.1, 0.1, 0.1, 0.1], "class_id": 0}
            ]
        }
        res = check_label_flipping(cand, ref)
        assert res["result"] == "ANOMALY_DETECTED"
        assert res["evidence"]["flip_count"] == 1
        assert res["evidence"]["object_level_breakdown"]["missing_objects"] == 1
        disc = res["evidence"]["examples"][0]
        assert disc["issue"] == "MISSING_ANNOTATION"
        assert disc["reference_class"] == 1

    def test_additional_annotation_detected(self):
        """Candidate having an extra bounding box not in reference."""
        ref = {
            "img1.jpg": [
                {"bbox": [0.1, 0.1, 0.1, 0.1], "class_id": 0}
            ]
        }
        cand = {
            "img1.jpg": [
                {"bbox": [0.1, 0.1, 0.1, 0.1], "class_id": 0},
                {"bbox": [0.7, 0.7, 0.1, 0.1], "class_id": 2},
            ]
        }
        res = check_label_flipping(cand, ref)
        assert res["result"] == "ANOMALY_DETECTED"
        assert res["evidence"]["flip_count"] == 1
        assert res["evidence"]["object_level_breakdown"]["additional_objects"] == 1
        disc = res["evidence"]["examples"][0]
        assert disc["issue"] == "ADDITIONAL_ANNOTATION"
        assert disc["candidate_class"] == 2

    def test_classification_multiset_matching(self):
        """When bounding boxes are absent, multiset matching applies."""
        ref = {
            "img1.jpg": [{"class_id": 0}, {"class_id": 1}, {"class_id": 1}]
        }
        cand_clean = {
            "img1.jpg": [{"class_id": 1}, {"class_id": 0}, {"class_id": 1}]
        }
        res_clean = check_label_flipping(cand_clean, ref)
        assert res_clean["result"] == "PASS"

        cand_flipped = {
            "img1.jpg": [{"class_id": 1}, {"class_id": 2}, {"class_id": 1}]
        }
        res_flipped = check_label_flipping(cand_flipped, ref)
        assert res_flipped["result"] == "ANOMALY_DETECTED"


# ---------------------------------------------------------------------------
# 3. Perceptual Hashing & Near-Duplicate Detection
# ---------------------------------------------------------------------------

class TestPerceptualHashing:
    def test_dhash_identical_images(self, tmp_path):
        img_a = Image.new("RGB", (64, 64), color=(100, 150, 200))
        img_path = tmp_path / "img_a.png"
        img_a.save(img_path)

        hash1 = compute_dhash(img_path)
        hash2 = compute_dhash(img_path)
        assert hash1 == hash2
        assert len(hash1) == 16  # 64 bits represented as 16 hex characters
        assert hamming_distance(hash1, hash2) == 0

    def test_dhash_near_duplicate_compression(self, tmp_path):
        """Recompressing a smooth image should have a very small Hamming distance."""
        x = np.linspace(0, 10, 128)
        y = np.linspace(0, 10, 128)
        xx, yy = np.meshgrid(x, y)
        arr = ((np.sin(xx) + np.cos(yy) + 2) * 60).astype(np.uint8)
        img = Image.fromarray(arr, mode="L").convert("RGB")
        png_path = tmp_path / "original.png"
        jpg_path = tmp_path / "recompressed.jpg"
        img.save(png_path)
        img.save(jpg_path, format="JPEG", quality=75)

        h_png = compute_dhash(png_path)
        h_jpg = compute_dhash(jpg_path)
        dist = hamming_distance(h_png, h_jpg)
        assert dist <= 4  # Minimal perceptual difference

    def test_dhash_different_images(self, tmp_path):
        """Horizontal vs vertical gradient should have large Hamming distance."""
        h_grad = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
        v_grad = np.tile(np.linspace(0, 255, 64, dtype=np.uint8).reshape((64, 1)), (1, 64))

        p1 = tmp_path / "h.png"
        p2 = tmp_path / "v.png"
        Image.fromarray(h_grad).save(p1)
        Image.fromarray(v_grad).save(p2)

        dist = hamming_distance(compute_dhash(p1), compute_dhash(p2))
        assert dist > 15

    def test_check_near_duplicates_detector(self, tmp_path):
        x = np.linspace(0, 10, 128)
        y = np.linspace(0, 10, 128)
        xx, yy = np.meshgrid(x, y)
        arr = ((np.sin(xx) + np.cos(yy) + 2) * 60).astype(np.uint8)
        img = Image.fromarray(arr, mode="L").convert("RGB")
        p1 = tmp_path / "img1.png"
        p2 = tmp_path / "img2.jpg"
        img.save(p1)
        img.save(p2, format="JPEG", quality=80)

        # Separate unrelated image
        arr_rand = np.random.RandomState(42).randint(0, 255, (128, 128, 3), dtype=np.uint8)
        p3 = tmp_path / "img3.png"
        Image.fromarray(arr_rand).save(p3)

        records = [
            {"filename": p1.name, "path": str(p1), "dhash": compute_dhash(p1)},
            {"filename": p2.name, "path": str(p2), "dhash": compute_dhash(p2)},
            {"filename": p3.name, "path": str(p3), "dhash": compute_dhash(p3)},
        ]
        res = check_near_duplicates(records, max_hamming_distance=4, threshold_pct=0.01)
        assert res["result"] == "ANOMALY_DETECTED"
        assert res["evidence"]["near_duplicate_pairs_count"] >= 1


# ---------------------------------------------------------------------------
# 4. Poisoning Detection: Reference Calibration & Masking Mitigation
# ---------------------------------------------------------------------------

class TestPoisoningDetectionRobust:
    def test_reference_calibrated_prevents_masking(self):
        """When 5 outliers are added, self-referenced Z-score standard deviation blows up,
        masking outliers. Reference-calibrated MAD with clean reference flags them."""
        # Clean reference distribution
        ref_records = [
            {"filename": f"ref_{i}.jpg", "pixel_stats": {
                "mean_r": 120.0 + np.sin(i), "mean_g": 100.0 + np.cos(i), "mean_b": 80.0,
                "std_r": 20.0, "std_g": 20.0, "std_b": 20.0
            }} for i in range(25)
        ]

        # Candidate dataset containing mostly clean samples plus 5 poisoned samples
        cand_clean = [
            {"filename": f"clean_{i}.jpg", "pixel_stats": {
                "mean_r": 120.0 + np.sin(i), "mean_g": 100.0 + np.cos(i), "mean_b": 80.0,
                "std_r": 20.0, "std_g": 20.0, "std_b": 20.0
            }} for i in range(20)
        ]
        # Injected poisoning: bright red tint
        cand_poison = [
            {"filename": f"poison_{i}.jpg", "pixel_stats": {
                "mean_r": 240.0, "mean_g": 30.0, "mean_b": 30.0,
                "std_r": 5.0, "std_g": 5.0, "std_b": 5.0
            }} for i in range(5)
        ]
        candidate_records = cand_clean + cand_poison

        # Run reference-calibrated robust MAD
        res_calib = check_data_poisoning(
            candidate_records,
            reference_records=ref_records,
            z_threshold=2.5,
            use_robust_mad=True
        )
        assert res_calib["result"] == "STATISTICAL_SHIFT"
        flagged_calib = [f["filename"] for f in res_calib["evidence"]["flagged_images"]]
        for i in range(5):
            assert f"poison_{i}.jpg" in flagged_calib
        assert len(res_calib["evidence"]["continuous_scores"]) == 25


# ---------------------------------------------------------------------------
# 5. OOD Detection: Reference Calibration & Regularization
# ---------------------------------------------------------------------------

class TestOODDetectionRobust:
    def test_reference_calibrated_ood(self):
        """OOD samples with different channel distribution flagged using reference calibration."""
        # 30 Clean reference samples: around RGB (120, 100, 80)
        rng = np.random.RandomState(42)
        ref_records = [
            {"filename": f"ref_{i}.jpg", "pixel_stats": {
                "mean_r": float(rng.normal(120, 5)),
                "mean_g": float(rng.normal(100, 5)),
                "mean_b": float(rng.normal(80, 5)),
                "std_r": float(rng.normal(20, 2)),
                "std_g": float(rng.normal(20, 2)),
                "std_b": float(rng.normal(20, 2)),
            }} for i in range(30)
        ]

        # Candidate dataset with clean in-distribution and 3 OOD samples
        cand_in = [
            {"filename": f"in_{i}.jpg", "pixel_stats": {
                "mean_r": float(rng.normal(120, 5)),
                "mean_g": float(rng.normal(100, 5)),
                "mean_b": float(rng.normal(80, 5)),
                "std_r": float(rng.normal(20, 2)),
                "std_g": float(rng.normal(20, 2)),
                "std_b": float(rng.normal(20, 2)),
            }} for i in range(15)
        ]
        cand_ood = [
            {"filename": f"ood_{i}.jpg", "pixel_stats": {
                "mean_r": float(rng.normal(30, 2)),
                "mean_g": float(rng.normal(200, 2)),
                "mean_b": float(rng.normal(220, 2)),
                "std_r": 5.0, "std_g": 5.0, "std_b": 5.0
            }} for i in range(3)
        ]
        cand_records = cand_in + cand_ood

        res = check_ood_samples(
            cand_records,
            reference_records=ref_records,
            threshold=3.5,
        )
        assert res["result"] == "ANOMALY_DETECTED"
        flagged = [f["filename"] for f in res["evidence"]["flagged_candidates"]]
        for i in range(3):
            assert f"ood_{i}.jpg" in flagged
        assert len(res["evidence"]["continuous_scores"]) == 18


# ---------------------------------------------------------------------------
# 6. Benchmark Data Integrity & Pipeline Smoke Test
# ---------------------------------------------------------------------------

class TestBenchmarkIntegrity:
    def test_benchmark_directories_exist(self):
        assert BENCHMARK_DIR.exists()
        assert (BENCHMARK_DIR / "reference").exists()
        assert (BENCHMARK_DIR / "calibration").exists()
        assert (BENCHMARK_DIR / "evaluation").exists()

    def test_ground_truth_isolation(self):
        """Check that ground truth files exist and specify ground truth without leakage."""
        calib_gt_path = BENCHMARK_DIR / "calibration" / "ground_truth.json"
        eval_gt_path = BENCHMARK_DIR / "evaluation" / "ground_truth.json"

        assert calib_gt_path.exists()
        assert eval_gt_path.exists()

        with open(eval_gt_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)

        assert len(gt_data) == 130

        # Verify ground truth metadata contains scenario and labels
        first_sample = next(iter(gt_data.values()))
        assert "is_anomaly" in first_sample
        assert "scenario" in first_sample


# ---------------------------------------------------------------------------
# 7. Phase 2.1 Independent Review Regression Tests
# ---------------------------------------------------------------------------

class TestPhase21RegressionFixes:
    def test_hungarian_colocated_boxes_class_tie_break(self):
        """Co-located overlapping boxes with distinct classes must not cross-match."""
        ref = {
            "img.jpg": [
                {"box": [0.2, 0.2, 0.6, 0.6], "class_id": 0},
                {"box": [0.2, 0.2, 0.6, 0.6], "class_id": 1},
            ]
        }
        cand = {
            "img.jpg": [
                {"box": [0.2, 0.2, 0.6, 0.6], "class_id": 1},
                {"box": [0.2, 0.2, 0.6, 0.6], "class_id": 0},
            ]
        }
        res = check_label_flipping(cand, ref)
        assert res["result"] == "PASS"
        assert res["evidence"]["flip_count"] == 0
        assert len(res["evidence"]["examples"]) == 0

    def test_coco_category_id_label_flipping(self):
        """Annotations with COCO category_id must be recognized and matched."""
        ref = {
            "img.jpg": [{"bbox": [10, 10, 50, 50], "category_id": 4}]
        }
        cand_clean = {
            "img.jpg": [{"bbox": [10, 10, 50, 50], "category_id": 4}]
        }
        cand_flipped = {
            "img.jpg": [{"bbox": [10, 10, 50, 50], "category_id": 9}]
        }
        res_clean = check_label_flipping(cand_clean, ref)
        assert res_clean["result"] == "PASS"

        res_flipped = check_label_flipping(cand_flipped, ref)
        assert res_flipped["result"] == "ANOMALY_DETECTED"
        assert res_flipped["evidence"]["object_level_breakdown"]["class_changes"] == 1

    def test_poisoning_zero_mad_fallback(self):
        """When reference samples have 0 MAD, tiny float deviations must not produce huge Z scores."""
        ref_records = [
            {"filename": f"ref_{i}.jpg", "pixel_stats": {
                "mean_r": 120.0, "mean_g": 100.0, "mean_b": 80.0,
                "std_r": 20.0, "std_g": 20.0, "std_b": 20.0
            }} for i in range(25)
        ]
        # Candidate with microscopic float rounding deviation (1e-4)
        cand_records = [
            {"filename": "micro_diff.jpg", "pixel_stats": {
                "mean_r": 120.0001, "mean_g": 100.0, "mean_b": 80.0,
                "std_r": 20.0, "std_g": 20.0, "std_b": 20.0
            }}
        ]
        res = check_data_poisoning(cand_records, reference_records=ref_records, use_robust_mad=True, z_threshold=3.0)
        assert res["result"] == "PASS"
        assert res["evidence"]["continuous_scores"]["micro_diff.jpg"] < 0.01

    def test_ood_covariance_regularization(self):
        """Collinear reference features must remain invertible with ridge regularization."""
        # Collinear R=G=B (grayscale images)
        ref_records = [
            {"filename": f"ref_{i}.jpg", "pixel_stats": {
                "mean_r": float(100 + i), "mean_g": float(100 + i), "mean_b": float(100 + i),
            }} for i in range(10)
        ]
        cand_records = [
            {"filename": "clean.jpg", "pixel_stats": {"mean_r": 105.0, "mean_g": 105.0, "mean_b": 105.0}},
            {"filename": "ood.jpg", "pixel_stats": {"mean_r": 250.0, "mean_g": 10.0, "mean_b": 10.0}},
        ]
        res = check_ood_samples(cand_records, reference_records=ref_records, regularization=1e-3, threshold=5.0)
        assert "clean.jpg" in res["evidence"]["continuous_scores"]
        assert "ood.jpg" in res["evidence"]["continuous_scores"]
        assert res["evidence"]["continuous_scores"]["clean.jpg"] < 2.0
        assert res["evidence"]["continuous_scores"]["ood.jpg"] > 5.0

    def test_trigger_patterns_small_resolution(self, tmp_path):
        """Images smaller than patch size should not crash and report clean."""
        tiny_path = tmp_path / "tiny.png"
        Image.new("RGB", (10, 10), (128, 128, 128)).save(tiny_path)
        res = check_trigger_patterns([tiny_path], patch_size=16)
        assert res["result"] == "PASS"
        assert len(res["evidence"]["flagged_images"]) == 0

