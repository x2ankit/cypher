"""
scripts/run_benchmark.py

Rigorous Scientific Benchmark Runner for CYPHER Dataset Assurance Subsystem.
Measures detection validity, statistical power, execution latency, and peak memory.

Key Protocols Enforced:
  - Zero experimental leakage: Thresholds selected on Calibration split and frozen.
  - Evaluation executed on independent Evaluation split.
  - Metrics computed independently per threat scenario:
      * Precision, Recall, F1, FPR, FNR
      * AUROC and AUPR for continuous anomaly scores
      * FPR at 95% TPR for OOD detection
      * Latency (mean, p95 in ms) and Peak Memory (tracemalloc in KiB)
      * Exact-match vs near-duplicate reported separately
      * Object-level vs image-level label integrity reported separately
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import time
import tracemalloc
import platform
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
BENCHMARK_DIR = ROOT / "data" / "benchmark"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

from src.dataset.inspector import inspect_yolo_dataset, inspect_image
from src.dataset.security_checks import (
    check_duplicate_flooding,
    check_near_duplicates,
    check_label_flipping,
    check_trigger_patterns,
    check_data_poisoning,
    check_ood_samples,
)

def compute_binary_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """Compute TP, FP, TN, FN, Precision, Recall, F1, FPR, FNR."""
    y_t = np.array(y_true, dtype=int)
    y_p = np.array(y_pred, dtype=int)
    
    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    tn = int(np.sum((y_t == 0) & (y_p == 0)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "fpr": round(float(fpr), 4),
        "fnr": round(float(fnr), 4),
    }

def compute_continuous_metrics(y_true: List[int], scores: List[float]) -> Dict[str, Optional[float]]:
    """Compute AUROC and AUPR."""
    y_t = np.array(y_true, dtype=int)
    s = np.array(scores, dtype=float)
    
    # Check if both classes are present
    if len(np.unique(y_t)) < 2:
        return {"auroc": float("nan"), "aupr": float("nan")}
    
    try:
        auroc = float(roc_auc_score(y_t, s))
    except Exception:
        auroc = float("nan")
        
    try:
        aupr = float(average_precision_score(y_t, s))
    except Exception:
        aupr = float("nan")
        
    return {
        "auroc": round(auroc, 4) if not np.isnan(auroc) else None,
        "aupr": round(aupr, 4) if not np.isnan(aupr) else None
    }

def compute_fpr_at_95_tpr(y_true: List[int], scores: List[float]) -> Optional[float]:
    """Compute False Positive Rate at >= 95% True Positive Rate."""
    y_t = np.array(y_true, dtype=int)
    s = np.array(scores, dtype=float)
    if len(np.unique(y_t)) < 2:
        return None
    
    # Sort distinct thresholds in descending order
    sorted_scores = np.sort(s)[::-1]
    best_fpr = 1.0
    found = False
    
    for thresh in sorted_scores:
        y_pred = (s >= thresh).astype(int)
        tp = np.sum((y_t == 1) & (y_pred == 1))
        fn = np.sum((y_t == 1) & (y_pred == 0))
        fp = np.sum((y_t == 0) & (y_pred == 1))
        tn = np.sum((y_t == 0) & (y_pred == 0))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        if tpr >= 0.95:
            best_fpr = min(best_fpr, fpr)
            found = True
            break
            
    return round(float(best_fpr), 4) if found else None

# ---------------------------------------------------------------------------
# Calibration Step (Freeze Thresholds)
# ---------------------------------------------------------------------------

def run_calibration(calib_dir: Path, ref_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Tune and freeze thresholds on calibration split using empirical data.
    """
    print("\n--- Running Calibration to Freeze Detection Thresholds ---")
    cal_gt = json.loads((calib_dir / "ground_truth.json").read_text(encoding="utf-8"))
    
    cal_res = inspect_yolo_dataset(calib_dir / "images", calib_dir / "labels")
    cal_records = cal_res["records"]
    
    ref_records = None
    if ref_dir and ref_dir.exists():
        ref_res = inspect_yolo_dataset(ref_dir / "images", ref_dir / "labels")
        ref_records = ref_res["records"]

    # 1. Calibrate Trigger Entropy Threshold
    img_paths = [Path(r["path"]) for r in cal_records if Path(r["path"]).exists()]
    trig_res = check_trigger_patterns(img_paths, patch_size=16, entropy_threshold=2.0)
    trig_scores = trig_res["evidence"]["continuous_scores"]
    clean_trig = [trig_scores[fn] for fn in trig_scores if cal_gt.get(fn, {}).get("scenario") == "clean"]
    anom_trig = [trig_scores[fn] for fn in trig_scores if cal_gt.get(fn, {}).get("threat_category") == "trigger"]
    
    min_clean_entropy = min(clean_trig) if clean_trig else 1.5
    max_anom_entropy = max(anom_trig) if anom_trig else 0.9
    calibrated_entropy_threshold = round(float(min(1.0, (min_clean_entropy + max_anom_entropy) / 2.0)), 2)
    print(f"  [Calibration] Trigger Entropy: clean_min={min_clean_entropy:.2f}, anom_max={max_anom_entropy:.2f} -> frozen_threshold={calibrated_entropy_threshold}")

    # 2. Calibrate Statistical Poisoning Threshold
    poison_res = check_data_poisoning(cal_records, reference_records=ref_records, use_robust_mad=True, z_threshold=2.5)
    p_scores = poison_res["evidence"]["continuous_scores"]
    clean_p = [p_scores[fn] for fn in p_scores if cal_gt.get(fn, {}).get("scenario") == "clean"]
    max_clean_z = max(clean_p) if clean_p else 3.0
    calibrated_z_threshold = round(float(max(3.0, np.ceil(max_clean_z * 10) / 10.0)), 1)
    print(f"  [Calibration] Poison Robust Z: clean_max={max_clean_z:.2f} -> frozen_threshold={calibrated_z_threshold}")

    # 3. Calibrate OOD Mahalanobis Threshold
    ood_res = check_ood_samples(cal_records, reference_records=ref_records)
    ood_scores = ood_res["evidence"]["continuous_scores"]
    clean_ood = [ood_scores[fn] for fn in ood_scores if cal_gt.get(fn, {}).get("scenario") == "clean"]
    max_clean_dist = max(clean_ood) if clean_ood else 3.0
    calibrated_ood_threshold = round(float(max_clean_dist * 1.5), 2)
    print(f"  [Calibration] OOD Mahalanobis: clean_max={max_clean_dist:.2f} -> frozen_threshold={calibrated_ood_threshold}")

    frozen_config = {
        "duplicate_exact_threshold_pct": 0.05,
        "dhash_max_hamming_distance": 8,
        "dhash_threshold_pct": 0.05,
        "label_flip_threshold": 0.0,
        "label_iou_match_threshold": 0.5,
        "label_iou_clean_threshold": 0.95,
        "trigger_entropy_threshold": calibrated_entropy_threshold,
        "trigger_patch_size": 16,
        "poison_z_threshold": calibrated_z_threshold,
        "ood_calibrated_threshold": calibrated_ood_threshold,
    }
    
    print(f"Frozen Calibration Config: {json.dumps(frozen_config, indent=2)}")
    return frozen_config

# ---------------------------------------------------------------------------
# Benchmark Execution Engine
# ---------------------------------------------------------------------------

def run_evaluation(eval_dir: Path, ref_dir: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    print("\n--- Executing Independent Evaluation Benchmark ---")
    eval_gt = json.loads((eval_dir / "ground_truth.json").read_text(encoding="utf-8"))
    ref_gt = json.loads((ref_dir / "ground_truth.json").read_text(encoding="utf-8"))
    ref_annotations_map = json.loads((eval_dir / "reference_annotations.json").read_text(encoding="utf-8"))

    # Inspect Reference & Evaluation Datasets
    t0 = time.perf_counter()
    ref_inspection = inspect_yolo_dataset(ref_dir / "images", ref_dir / "labels")
    eval_inspection = inspect_yolo_dataset(eval_dir / "images", eval_dir / "labels")
    inspection_time = (time.perf_counter() - t0) * 1000

    eval_records = eval_inspection["records"]
    ref_records = ref_inspection["records"]
    candidate_annotations = {r["filename"]: r.get("annotations", []) for r in eval_records}

    results = {}

    # =======================================================================
    # 1. Exact Duplicate Detection (SHA-256)
    # =======================================================================
    tracemalloc.start()
    t_start = time.perf_counter()
    dup_res = check_duplicate_flooding(eval_records, threshold_pct=config["duplicate_exact_threshold_pct"])
    dup_latency_ms = (time.perf_counter() - t_start) * 1000
    _, dup_peak_kb = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Exact duplicate performance: True anomalies are exact duplicates
    exact_dup_y_true = []
    exact_dup_y_pred = []
    flagged_exact_files = set()
    for grp in dup_res["evidence"]["duplicate_groups"]:
        for f in grp["files"]:
            flagged_exact_files.add(f)

    for fn, info in eval_gt.items():
        if info["threat_category"] in ("clean", "duplicate"):
            is_exact = (info.get("scenario") == "exact_duplicate")
            exact_dup_y_true.append(1 if is_exact else 0)
            exact_dup_y_pred.append(1 if fn in flagged_exact_files else 0)

    # Near duplicate performance on exact detector (should fail/miss them)
    all_dup_y_true = []
    all_dup_y_pred = []
    for fn, info in eval_gt.items():
        if info["threat_category"] in ("clean", "duplicate"):
            is_any_dup = info["is_anomaly"]
            all_dup_y_true.append(1 if is_any_dup else 0)
            all_dup_y_pred.append(1 if fn in flagged_exact_files else 0)

    results["duplicate_exact_sha256"] = {
        "description": "Baseline exact byte SHA-256 duplicate detection",
        "latency_ms": round(dup_latency_ms, 2),
        "peak_memory_kb": round(dup_peak_kb / 1024, 2),
        "exact_duplicates_only": compute_binary_metrics(exact_dup_y_true, exact_dup_y_pred),
        "all_duplicates_including_recompressed": compute_binary_metrics(all_dup_y_true, all_dup_y_pred),
    }

    # =======================================================================
    # 2. Perceptual Near-Duplicate Detection (dHash)
    # =======================================================================
    tracemalloc.start()
    t_start = time.perf_counter()
    near_dup_res = check_near_duplicates(
        eval_records,
        max_hamming_distance=config["dhash_max_hamming_distance"],
        threshold_pct=config["dhash_threshold_pct"]
    )
    near_dup_latency_ms = (time.perf_counter() - t_start) * 1000
    _, near_dup_peak_kb = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    flagged_near_files = set()
    for pair in near_dup_res["evidence"]["pairs"]:
        flagged_near_files.add(pair["file_a"])
        flagged_near_files.add(pair["file_b"])

    near_dup_y_true = []
    near_dup_y_pred = []
    dhash_scores = []  # continuous score: (64 - min_distance)
    for fn, info in eval_gt.items():
        if info["threat_category"] in ("clean", "duplicate"):
            near_dup_y_true.append(1 if info["is_anomaly"] else 0)
            near_dup_y_pred.append(1 if fn in flagged_near_files else 0)
            # Find min hamming distance for this file
            min_d = 64
            for pair in near_dup_res["evidence"]["pairs"]:
                if pair["file_a"] == fn or pair["file_b"] == fn:
                    min_d = min(min_d, pair["hamming_distance"])
            dhash_scores.append(float(64 - min_d))

    dhash_binary = compute_binary_metrics(near_dup_y_true, near_dup_y_pred)
    dhash_cont = compute_continuous_metrics(near_dup_y_true, dhash_scores)

    results["duplicate_perceptual_dhash"] = {
        "description": "Perceptual difference hashing (dHash, max Hamming dist 8)",
        "latency_ms": round(near_dup_latency_ms, 2),
        "peak_memory_kb": round(near_dup_peak_kb / 1024, 2),
        "binary_metrics": dhash_binary,
        "continuous_metrics": dhash_cont,
    }

    # =======================================================================
    # 3. Label Flipping & Integrity (Hungarian IoU Matching)
    # =======================================================================
    tracemalloc.start()
    t_start = time.perf_counter()
    label_res = check_label_flipping(
        candidate_annotations=candidate_annotations,
        reference_annotations=ref_annotations_map,
        flip_threshold=config["label_flip_threshold"],
        iou_match_threshold=config["label_iou_match_threshold"],
        iou_clean_threshold=config["label_iou_clean_threshold"],
    )
    label_latency_ms = (time.perf_counter() - t_start) * 1000
    _, label_peak_kb = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    flagged_label_files = set(ex["file"] for ex in label_res["evidence"]["examples"])

    # Image-level label integrity metrics
    label_img_y_true = []
    label_img_y_pred = []
    for fn, info in eval_gt.items():
        if info["threat_category"] in ("clean", "annotation"):
            label_img_y_true.append(1 if info["is_anomaly"] else 0)
            label_img_y_pred.append(1 if fn in flagged_label_files else 0)

    label_img_metrics = compute_binary_metrics(label_img_y_true, label_img_y_pred)

    results["label_integrity_hungarian"] = {
        "description": "Optimal bipartite matching (Hungarian IoU) for label integrity",
        "latency_ms": round(label_latency_ms, 2),
        "peak_memory_kb": round(label_peak_kb / 1024, 2),
        "image_level_metrics": label_img_metrics,
        "object_level_breakdown": label_res["evidence"]["object_level_breakdown"],
        "reordered_annotations_false_positives": sum(
            1 for fn in flagged_label_files if eval_gt[fn].get("scenario") == "reordered_clean"
        ),
    }

    # =======================================================================
    # 4. Trigger Pattern Detection (Patch Entropy)
    # =======================================================================
    eval_image_paths = [Path(r["path"]) for r in eval_records if Path(r["path"]).exists()]
    tracemalloc.start()
    t_start = time.perf_counter()
    trig_res = check_trigger_patterns(
        eval_image_paths,
        patch_size=config["trigger_patch_size"],
        entropy_threshold=config["trigger_entropy_threshold"],
    )
    trig_latency_ms = (time.perf_counter() - t_start) * 1000
    _, trig_peak_kb = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    flagged_trig_files = set(f["filename"] for f in trig_res["evidence"]["flagged_images"])
    trig_continuous = trig_res["evidence"]["continuous_scores"]

    trig_y_true = []
    trig_y_pred = []
    trig_scores = []  # continuous score: (8.0 - min_entropy), higher is more anomalous
    for fn, info in eval_gt.items():
        if info["threat_category"] in ("clean", "trigger"):
            trig_y_true.append(1 if info["is_anomaly"] else 0)
            trig_y_pred.append(1 if fn in flagged_trig_files else 0)
            min_ent = trig_continuous.get(fn, 8.0)
            trig_scores.append(8.0 - min_ent)

    trig_binary = compute_binary_metrics(trig_y_true, trig_y_pred)
    trig_cont = compute_continuous_metrics(trig_y_true, trig_scores)

    results["trigger_patch_entropy"] = {
        "description": "Sliding window Shannon patch entropy (16x16 window, threshold < 1.0)",
        "latency_ms": round(trig_latency_ms, 2),
        "peak_memory_kb": round(trig_peak_kb / 1024, 2),
        "binary_metrics": trig_binary,
        "continuous_metrics": trig_cont,
    }

    # =======================================================================
    # 5. Statistical Poisoning Detection (Z-Score & Robust MAD)
    # =======================================================================
    # A. Baseline: Self-referenced Z-Score
    tracemalloc.start()
    t_start = time.perf_counter()
    poison_self_res = check_data_poisoning(
        eval_records,
        z_threshold=config["poison_z_threshold"],
        reference_records=None,
        use_robust_mad=False
    )
    poison_self_latency_ms = (time.perf_counter() - t_start) * 1000
    _, poison_self_peak_kb = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    flagged_poison_self = set(f["filename"] for f in poison_self_res["evidence"]["flagged_images"])
    poison_self_continuous = poison_self_res["evidence"]["continuous_scores"]

    # B. Improved: Reference-Calibrated Robust MAD
    tracemalloc.start()
    t_start = time.perf_counter()
    poison_ref_res = check_data_poisoning(
        eval_records,
        z_threshold=config["poison_z_threshold"],
        reference_records=ref_records,
        use_robust_mad=True
    )
    poison_ref_latency_ms = (time.perf_counter() - t_start) * 1000
    _, poison_ref_peak_kb = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    flagged_poison_ref = set(f["filename"] for f in poison_ref_res["evidence"]["flagged_images"])
    poison_ref_continuous = poison_ref_res["evidence"]["continuous_scores"]

    poison_y_true = []
    poison_self_y_pred = []
    poison_self_scores = []
    poison_ref_y_pred = []
    poison_ref_scores = []

    for fn, info in eval_gt.items():
        if info["threat_category"] in ("clean", "statistical"):
            poison_y_true.append(1 if info["is_anomaly"] else 0)
            poison_self_y_pred.append(1 if fn in flagged_poison_self else 0)
            poison_self_scores.append(poison_self_continuous.get(fn, 0.0))
            poison_ref_y_pred.append(1 if fn in flagged_poison_ref else 0)
            poison_ref_scores.append(poison_ref_continuous.get(fn, 0.0))

    results["poisoning_self_referenced_zscore"] = {
        "description": "Baseline self-referenced RGB Z-score outlier detection",
        "latency_ms": round(poison_self_latency_ms, 2),
        "peak_memory_kb": round(poison_self_peak_kb / 1024, 2),
        "binary_metrics": compute_binary_metrics(poison_y_true, poison_self_y_pred),
        "continuous_metrics": compute_continuous_metrics(poison_y_true, poison_self_scores),
    }

    results["poisoning_reference_calibrated_mad"] = {
        "description": "Improved reference-calibrated robust Modified Z-score (Median/MAD)",
        "latency_ms": round(poison_ref_latency_ms, 2),
        "peak_memory_kb": round(poison_ref_peak_kb / 1024, 2),
        "binary_metrics": compute_binary_metrics(poison_y_true, poison_ref_y_pred),
        "continuous_metrics": compute_continuous_metrics(poison_y_true, poison_ref_scores),
    }

    # =======================================================================
    # 6. Out-of-Distribution (OOD) Detection (Mahalanobis Distance)
    # =======================================================================
    # A. Baseline: Self-referenced Mahalanobis
    tracemalloc.start()
    t_start = time.perf_counter()
    ood_self_res = check_ood_samples(eval_records, reference_records=None)
    ood_self_latency_ms = (time.perf_counter() - t_start) * 1000
    _, ood_self_peak_kb = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    flagged_ood_self = set(f["filename"] for f in ood_self_res["evidence"].get("flagged_candidates", []))
    ood_self_continuous = ood_self_res["evidence"].get("continuous_scores", {})

    # B. Improved: Reference-calibrated Mahalanobis
    tracemalloc.start()
    t_start = time.perf_counter()
    ood_ref_res = check_ood_samples(
        eval_records,
        reference_records=ref_records,
        threshold=config["ood_calibrated_threshold"]
    )
    ood_ref_latency_ms = (time.perf_counter() - t_start) * 1000
    _, ood_ref_peak_kb = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    flagged_ood_ref = set(f["filename"] for f in ood_ref_res["evidence"].get("flagged_candidates", []))
    ood_ref_continuous = ood_ref_res["evidence"].get("continuous_scores", {})

    ood_y_true = []
    ood_self_y_pred = []
    ood_self_scores = []
    ood_ref_y_pred = []
    ood_ref_scores = []

    for fn, info in eval_gt.items():
        if info["threat_category"] in ("clean", "ood"):
            ood_y_true.append(1 if info["is_anomaly"] else 0)
            ood_self_y_pred.append(1 if fn in flagged_ood_self else 0)
            ood_self_scores.append(ood_self_continuous.get(fn, 0.0))
            ood_ref_y_pred.append(1 if fn in flagged_ood_ref else 0)
            ood_ref_scores.append(ood_ref_continuous.get(fn, 0.0))

    results["ood_self_referenced_mahalanobis"] = {
        "description": "Baseline self-referenced channel Mahalanobis distance",
        "latency_ms": round(ood_self_latency_ms, 2),
        "peak_memory_kb": round(ood_self_peak_kb / 1024, 2),
        "binary_metrics": compute_binary_metrics(ood_y_true, ood_self_y_pred),
        "continuous_metrics": compute_continuous_metrics(ood_y_true, ood_self_scores),
        "fpr_at_95_tpr": compute_fpr_at_95_tpr(ood_y_true, ood_self_scores),
    }

    results["ood_reference_calibrated_mahalanobis"] = {
        "description": "Improved reference-calibrated Mahalanobis distance",
        "latency_ms": round(ood_ref_latency_ms, 2),
        "peak_memory_kb": round(ood_ref_peak_kb / 1024, 2),
        "binary_metrics": compute_binary_metrics(ood_y_true, ood_ref_y_pred),
        "continuous_metrics": compute_continuous_metrics(ood_y_true, ood_ref_scores),
        "fpr_at_95_tpr": compute_fpr_at_95_tpr(ood_y_true, ood_ref_scores),
    }

    # Assemble complete benchmark report
    benchmark_report = {
        "benchmark_metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "platform": platform.platform(),
            "python_version": sys.version,
            "evaluation_samples_count": len(eval_gt),
            "reference_samples_count": len(ref_gt),
            "frozen_config": config,
        },
        "detector_benchmarks": results
    }

    out_file = REPORTS_DIR / "benchmark_results.json"
    out_file.write_text(json.dumps(benchmark_report, indent=2), encoding="utf-8")
    print(f"\n[OK] Benchmark completed successfully. Results saved to {out_file}")
    return benchmark_report


def print_markdown_summary(report: Dict[str, Any]):
    print("\n" + "=" * 80)
    print("CYPHER — EMPIRICAL BENCHMARK RESULTS")
    print("=" * 80)
    
    detectors = report["detector_benchmarks"]
    
    print("\n### 1. Detection Performance (Precision, Recall, F1, FPR, FNR, AUROC, AUPR)")
    print("| Detector | Target Threat | Precision | Recall | F1 | FPR | FNR | AUROC | AUPR | Latency (ms) | Peak RAM (KiB) |")
    print("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    # 1. Exact Duplicates
    d_exact = detectors["duplicate_exact_sha256"]["exact_duplicates_only"]
    print(f"| SHA-256 Duplicate (Exact Only) | Exact Copies | {d_exact['precision']} | {d_exact['recall']} | {d_exact['f1']} | {d_exact['fpr']} | {d_exact['fnr']} | N/A | N/A | {detectors['duplicate_exact_sha256']['latency_ms']} | {detectors['duplicate_exact_sha256']['peak_memory_kb']} |")

    d_all = detectors["duplicate_exact_sha256"]["all_duplicates_including_recompressed"]
    print(f"| SHA-256 Duplicate (All Dups) | All Dups (Near+Exact) | {d_all['precision']} | {d_all['recall']} | {d_all['f1']} | {d_all['fpr']} | {d_all['fnr']} | N/A | N/A | {detectors['duplicate_exact_sha256']['latency_ms']} | {detectors['duplicate_exact_sha256']['peak_memory_kb']} |")

    # 2. Perceptual dHash
    d_p = detectors["duplicate_perceptual_dhash"]["binary_metrics"]
    d_pc = detectors["duplicate_perceptual_dhash"]["continuous_metrics"]
    print(f"| dHash Near-Duplicate (Improved) | Near-Duplicates | {d_p['precision']} | {d_p['recall']} | {d_p['f1']} | {d_p['fpr']} | {d_p['fnr']} | {d_pc['auroc']} | {d_pc['aupr']} | {detectors['duplicate_perceptual_dhash']['latency_ms']} | {detectors['duplicate_perceptual_dhash']['peak_memory_kb']} |")

    # 3. Label Integrity
    d_l = detectors["label_integrity_hungarian"]["image_level_metrics"]
    print(f"| Hungarian IoU Label Match | Label Alterations | {d_l['precision']} | {d_l['recall']} | {d_l['f1']} | {d_l['fpr']} | {d_l['fnr']} | N/A | N/A | {detectors['label_integrity_hungarian']['latency_ms']} | {detectors['label_integrity_hungarian']['peak_memory_kb']} |")

    # 4. Trigger Entropy
    d_t = detectors["trigger_patch_entropy"]["binary_metrics"]
    d_tc = detectors["trigger_patch_entropy"]["continuous_metrics"]
    print(f"| Patch Shannon Entropy | Backdoor Triggers | {d_t['precision']} | {d_t['recall']} | {d_t['f1']} | {d_t['fpr']} | {d_t['fnr']} | {d_tc['auroc']} | {d_tc['aupr']} | {detectors['trigger_patch_entropy']['latency_ms']} | {detectors['trigger_patch_entropy']['peak_memory_kb']} |")

    # 5. Poisoning Self vs Ref
    d_ps = detectors["poisoning_self_referenced_zscore"]["binary_metrics"]
    d_psc = detectors["poisoning_self_referenced_zscore"]["continuous_metrics"]
    print(f"| Z-Score Poison (Self-Ref) | Statistical Outliers | {d_ps['precision']} | {d_ps['recall']} | {d_ps['f1']} | {d_ps['fpr']} | {d_ps['fnr']} | {d_psc['auroc']} | {d_psc['aupr']} | {detectors['poisoning_self_referenced_zscore']['latency_ms']} | {detectors['poisoning_self_referenced_zscore']['peak_memory_kb']} |")

    d_pr = detectors["poisoning_reference_calibrated_mad"]["binary_metrics"]
    d_prc = detectors["poisoning_reference_calibrated_mad"]["continuous_metrics"]
    print(f"| Robust MAD (Reference-Calib) | Statistical Outliers | {d_pr['precision']} | {d_pr['recall']} | {d_pr['f1']} | {d_pr['fpr']} | {d_pr['fnr']} | {d_prc['auroc']} | {d_prc['aupr']} | {detectors['poisoning_reference_calibrated_mad']['latency_ms']} | {detectors['poisoning_reference_calibrated_mad']['peak_memory_kb']} |")

    # 6. OOD Self vs Ref
    d_os = detectors["ood_self_referenced_mahalanobis"]["binary_metrics"]
    d_osc = detectors["ood_self_referenced_mahalanobis"]["continuous_metrics"]
    print(f"| Mahalanobis OOD (Self-Ref) | Out-of-Distribution | {d_os['precision']} | {d_os['recall']} | {d_os['f1']} | {d_os['fpr']} | {d_os['fnr']} | {d_osc['auroc']} | {d_osc['aupr']} | {detectors['ood_self_referenced_mahalanobis']['latency_ms']} | {detectors['ood_self_referenced_mahalanobis']['peak_memory_kb']} |")

    d_or = detectors["ood_reference_calibrated_mahalanobis"]["binary_metrics"]
    d_orc = detectors["ood_reference_calibrated_mahalanobis"]["continuous_metrics"]
    print(f"| Mahalanobis OOD (Ref-Calib) | Out-of-Distribution | {d_or['precision']} | {d_or['recall']} | {d_or['f1']} | {d_or['fpr']} | {d_or['fnr']} | {d_orc['auroc']} | {d_orc['aupr']} | {detectors['ood_reference_calibrated_mahalanobis']['latency_ms']} | {detectors['ood_reference_calibrated_mahalanobis']['peak_memory_kb']} |")

    print("\n### 2. OOD Specialized Metrics (FPR at 95% TPR)")
    print(f"- **Self-Referenced Mahalanobis**: FPR @ 95% TPR = {detectors['ood_self_referenced_mahalanobis']['fpr_at_95_tpr']}")
    print(f"- **Reference-Calibrated Mahalanobis**: FPR @ 95% TPR = {detectors['ood_reference_calibrated_mahalanobis']['fpr_at_95_tpr']}")

    print("\n### 3. Label Integrity Matching Invariance")
    print(f"- **False positives on permuted/reordered clean annotations**: {detectors['label_integrity_hungarian']['reordered_annotations_false_positives']} (Optimal Hungarian Matching)")
    print(f"- **Object-level breakdown**: {json.dumps(detectors['label_integrity_hungarian']['object_level_breakdown'])}")


if __name__ == "__main__":
    ref_d = BENCHMARK_DIR / "reference"
    cal_d = BENCHMARK_DIR / "calibration"
    eval_d = BENCHMARK_DIR / "evaluation"

    if not (eval_d / "ground_truth.json").exists():
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_benchmark.py")], check=True)

    cfg = run_calibration(cal_d, ref_d)
    rep = run_evaluation(eval_d, ref_d, cfg)
    print_markdown_summary(rep)
