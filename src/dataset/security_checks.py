"""
src/dataset/security_checks.py

Reproducible security-oriented tests on training datasets.
Each function takes dataset inspection output or raw paths and returns
a structured finding dict with result, evidence, severity, confidence,
and recommended action.

Supported threat scenarios:
  - Poisoning detection (statistical outlier in pixel distribution)
  - Label flipping detection (annotation set comparison vs reference)
  - Duplicate flooding detection (hash-based exact duplicate count)
  - Trigger pattern insertion (uniform patch entropy search)
  - Out-of-distribution (OOD) insertion (Mahalanobis, already in inspector)

Limitations:
  - Statistical methods produce probabilistic signals, not definitive verdicts.
  - Trigger detection searches for low-entropy patches; sophisticated triggers
    may evade this heuristic.
  - Label flipping detection requires a known-good reference annotation set.
  - Does NOT claim to detect all attacks or confirm any attack with certainty.
"""

import hashlib
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

try:
    from src.api.audit_db import log_event
except ImportError:
    log_event = None

logger = logging.getLogger(__name__)

SEVERITY_LEVELS = ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_rgb(path: Path) -> Optional[np.ndarray]:
    try:
        with Image.open(path) as img:
            return np.array(img.convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        logger.warning("Cannot open image %s: %s", path, exc)
        return None


def _patch_entropy(patch: np.ndarray) -> float:
    """Shannon entropy of pixel value histogram across all channels."""
    flat = patch.reshape(-1)
    counts = np.bincount(flat, minlength=256)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_dhash(img: Union[Image.Image, str, Path], hash_size: int = 8) -> str:
    """
    Compute Difference Hash (dHash) for perceptual image comparison.
    Resizes to (hash_size + 1, hash_size), converts to grayscale, and compares
    adjacent horizontal pixels to yield a 64-bit fingerprint.
    """
    if isinstance(img, (str, Path)):
        with Image.open(img) as opened:
            return compute_dhash(opened, hash_size)
    resized = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    arr = np.array(resized, dtype=np.float32)
    diff = arr[:, 1:] > arr[:, :-1]
    decimal_value = 0
    hex_string = []
    for index, value in enumerate(diff.flatten()):
        if value:
            decimal_value += 2 ** (index % 8)
        if (index % 8) == 7:
            hex_string.append(hex(decimal_value)[2:].rjust(2, '0'))
            decimal_value = 0
    return "".join(hex_string)


def hamming_distance(h1: str, h2: str) -> int:
    """Calculate Hamming distance between two hex string hashes."""
    try:
        n1 = int(h1, 16)
        n2 = int(h2, 16)
        return bin(n1 ^ n2).count('1')
    except (ValueError, TypeError):
        return 64


def _extract_bbox(ann: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    """
    Extract bounding box in standard (x1, y1, x2, y2) coordinates.
    Supports YOLO normalized [x_center, y_center, width, height],
    COCO pixel [x, y, w, h], and explicit box [x1, y1, x2, y2].
    """
    if "bbox" in ann:
        b = ann["bbox"]
        if isinstance(b, (list, tuple)) and len(b) >= 4:
            x, y, w, h = float(b[0]), float(b[1]), float(b[2]), float(b[3])
            return (x, y, x + w, y + h)
    if "x_center" in ann and "width" in ann:
        xc = float(ann["x_center"])
        yc = float(ann["y_center"])
        w = float(ann["width"])
        h = float(ann["height"])
        return (xc - w / 2.0, yc - h / 2.0, xc + w / 2.0, yc + h / 2.0)
    if "box" in ann:
        b = ann["box"]
        if isinstance(b, (list, tuple)) and len(b) >= 4:
            return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    return None


def compute_bbox_iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes in (x1, y1, x2, y2).
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    if union <= 1e-9:
        return 0.0
    return float(intersection / union)


# ---------------------------------------------------------------------------
# 1. Duplicate flooding check (Exact SHA-256 and Perceptual dHash)
# ---------------------------------------------------------------------------

def check_duplicate_flooding(
    records: List[Dict[str, Any]],
    threshold_pct: float = 0.05,
    audit_id: str = "",
) -> Dict[str, Any]:
    """
    Detect exact-duplicate images via SHA-256 hash grouping.

    Args:
        records: list of per-image records from inspector.inspect_yolo_dataset
        threshold_pct: fraction of dataset that can be duplicates before flagging

    Returns:
        Finding dict with result, severity, evidence, confidence, recommended_action.
    """
    hash_map: Dict[str, List[str]] = {}
    for rec in records:
        sha = rec.get("sha256")
        if sha and sha != "NOT ASSESSED":
            hash_map.setdefault(sha, []).append(rec["filename"])

    dup_groups = {k: v for k, v in hash_map.items() if len(v) > 1}
    total = len(records)
    dup_count = sum(len(v) - 1 for v in dup_groups.values())
    dup_pct = dup_count / total if total > 0 else 0.0

    result = "PASS"
    severity = "INFO"
    if dup_pct >= threshold_pct:
        result = "ANOMALY_DETECTED"
        severity = "HIGH" if dup_pct >= 0.25 else "MEDIUM"
        
    if audit_id and log_event:
        log_event(audit_id, "Duplicate Flooding", "Excess Duplicate Rate", 
                  {"total": total, "dup_count": dup_count}, 
                  "rate = dup_count / total", 
                  {"rate": dup_pct, "threshold": threshold_pct}, 
                  f"{dup_pct:.4f}", f"{threshold_pct:.4f}", 
                  result, "", 0, "WARNING" if result != "PASS" else "PASS")

    return {
        "check_id": "SEC-DS-001",
        "name": "Duplicate Flooding Detection",
        "result": result,
        "severity": severity,
        "confidence": "HIGH",  # hash comparison is deterministic
        "evidence": {
            "total_images": total,
            "duplicate_count": dup_count,
            "duplicate_fraction": round(dup_pct, 4),
            "threshold_fraction": threshold_pct,
            "duplicate_groups": [
                {"sha256_prefix": k[:16] + "...", "files": v}
                for k, v in list(dup_groups.items())[:20]
            ],
        },
        "description": (
            f"{dup_count} exact duplicate image(s) detected ({dup_pct*100:.1f}% of dataset). "
            "Exact-duplicate flooding can bias model training. "
            "This is a deterministic finding based on SHA-256 collision."
        ),
        "recommended_action": (
            "Remove duplicate images and their corresponding labels before training. "
            "Investigate the provenance of duplicates for potential data poisoning intent."
        ),
        "limitation": (
            "Detects only exact byte-for-byte duplicates. Near-duplicates, "
            "perceptually similar images, or watermarked copies will NOT be detected."
        ),
        "trace": {
            "operation_id": "SHA256_DUPLICATE_CHECK",
            "operation_type": "Exact Hash Comparison",
            "formula": "duplicate_fraction = samples_in_duplicate_groups / total_samples",
            "inputs": {
                "total_samples": total,
                "samples_in_duplicate_groups": dup_count
            },
            "intermediate_values": {
                "duplicate_groups": [{"hash": k, "count": len(v)} for k, v in list(dup_groups.items())[:5]]
            },
            "result": round(dup_pct, 4),
            "threshold": threshold_pct,
            "comparison": f"{dup_pct:.4f} >= {threshold_pct}",
            "decision": result,
            "explanation": "Calculates the fraction of identical image bytes via SHA-256 digest equality. Zero duplicates correctly produce 0.0 fraction."
        }
    }


def check_near_duplicates(
    records: List[Dict[str, Any]],
    max_hamming_distance: int = 8,
    threshold_pct: float = 0.05,
    audit_id: str = "",
) -> Dict[str, Any]:
    """
    Detect perceptual near-duplicate images using Difference Hashing (dHash).
    Identifies recompressed images (JPEG), resized copies, crops, and slight color alterations
    that bypass exact SHA-256 checks.
    """
    hashes: List[Tuple[str, str]] = []
    for rec in records:
        fn = rec.get("filename", "")
        if "dhash" in rec and rec["dhash"]:
            hashes.append((fn, rec["dhash"]))
            continue
        path_str = rec.get("path") or rec.get("filepath")
        if path_str and Path(path_str).exists():
            try:
                h = compute_dhash(path_str)
                hashes.append((fn, h))
            except Exception:
                pass

    total = len(records)
    near_dup_pairs: List[Dict[str, Any]] = []
    flagged_files: set = set()

    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            f1, h1 = hashes[i]
            f2, h2 = hashes[j]
            dist = hamming_distance(h1, h2)
            if dist <= max_hamming_distance:
                flagged_files.add(f1)
                flagged_files.add(f2)
                near_dup_pairs.append({
                    "file_a": f1,
                    "file_b": f2,
                    "hamming_distance": dist,
                    "similarity": round(1.0 - (dist / 64.0), 4)
                })

    near_dup_count = len(flagged_files)
    near_dup_pct = near_dup_count / total if total > 0 else 0.0

    result = "PASS"
    severity = "INFO"
    if near_dup_pct >= threshold_pct and len(near_dup_pairs) > 0:
        result = "ANOMALY_DETECTED"
        severity = "HIGH" if near_dup_pct >= 0.25 else "MEDIUM"

    return {
        "check_id": "SEC-DS-001b",
        "name": "Perceptual Near-Duplicate Detection (dHash)",
        "result": result,
        "severity": severity,
        "confidence": "HIGH",
        "evidence": {
            "total_images": total,
            "near_duplicate_files_count": near_dup_count,
            "near_duplicate_pairs_count": len(near_dup_pairs),
            "near_duplicate_fraction": round(near_dup_pct, 4),
            "max_hamming_distance": max_hamming_distance,
            "pairs": near_dup_pairs[:20],
        },
        "description": (
            f"{len(near_dup_pairs)} perceptual near-duplicate pair(s) detected across {near_dup_count} images "
            f"({near_dup_pct*100:.1f}% of dataset, Hamming distance <= {max_hamming_distance})."
            if near_dup_pairs else
            f"Zero near-duplicate images detected across {total} images."
        ),
        "recommended_action": (
            "Deduplicate near-identical images to prevent validation leakage and sample bias."
            if near_dup_pairs else "No action required."
        ),
        "limitation": (
            "Perceptual hashing detects structural/gradient similarities. Significant crops (>30%) "
            "or severe occlusions may exceed the Hamming distance threshold."
        ),
    }


# ---------------------------------------------------------------------------
# 2. Label flipping & integrity detection (vs reference with Hungarian IoU matching)
# ---------------------------------------------------------------------------

def check_label_flipping(
    candidate_annotations: Dict[str, List[Dict]],
    reference_annotations: Optional[Dict[str, List[Dict]]] = None,
    flip_threshold: float = 0.0,
    iou_match_threshold: float = 0.5,
    iou_clean_threshold: float = 0.95,
) -> Dict[str, Any]:
    """
    Compare candidate annotations against a known-good reference set using
    optimal bipartite matching (Hungarian assignment) on bounding box IoU.

    Handles:
      - Bounding box coordinate shifts and modifications
      - Class ID changes for matched bounding boxes (label flips)
      - Missing objects (deleted annotations)
      - Additional objects (spurious annotations)
      - Arbitrary annotation list reordering (order-invariant matching)
      - Empty candidate or reference annotation lists
      - Classification labels without spatial bounding boxes (multiset matching)

    Args:
        candidate_annotations: {filename: [annotation_dicts]}  (candidate)
        reference_annotations: {filename: [annotation_dicts]}  (trusted reference)
        flip_threshold: fraction of mismatches that constitutes an overall anomaly
        iou_match_threshold: minimum IoU to consider candidate and reference the same object (default 0.5)
        iou_clean_threshold: minimum IoU to consider bounding box unmodified (default 0.95)

    Returns:
        Structured finding dict with object-level and image-level metrics.
    """
    if reference_annotations is None or not reference_annotations:
        return {
            "check_id": "SEC-DS-002",
            "name": "Label Flipping Detection",
            "result": "NOT_ASSESSED",
            "severity": "INFO",
            "confidence": "NOT_ASSESSED",
            "evidence": {
                "reason": "Reference annotations not provided or empty."
            },
            "description": "Label flipping detection requires a trusted reference annotation set. None was provided.",
            "recommended_action": "Provide reference annotations to evaluate label flipping.",
            "limitation": "Cannot detect label flips without ground truth reference annotations.",
        }

    common_files = sorted(set(candidate_annotations) & set(reference_annotations))
    missing_files = sorted(set(reference_annotations) - set(candidate_annotations))
    extra_files = sorted(set(candidate_annotations) - set(reference_annotations))

    anomalies: List[Dict[str, Any]] = []
    
    # Detailed counts
    obj_class_changes = 0
    obj_bbox_modifications = 0
    obj_missing = 0
    obj_additional = 0
    obj_matched_clean = 0

    img_clean_count = 0
    img_anomaly_count = 0

    # Image-level missing files
    for fname in missing_files:
        ref_count = len(reference_annotations[fname])
        obj_missing += ref_count
        anomalies.append({
            "file": fname,
            "issue": "MISSING_IMAGE_ANNOTATIONS",
            "missing_objects_count": ref_count,
        })
        img_anomaly_count += 1

    # Note unreferenced extra files in evidence without failing label check
    # (candidate datasets often contain newer images not yet in reference)

    for fname in common_files:
        cand_list = candidate_annotations[fname]
        ref_list = reference_annotations[fname]

        img_has_anomaly = False

        # Extract bboxes
        cand_boxes = [_extract_bbox(a) for a in cand_list]
        ref_boxes = [_extract_bbox(a) for a in ref_list]

        has_cand_boxes = len(cand_boxes) > 0 and all(b is not None for b in cand_boxes)
        has_ref_boxes = len(ref_boxes) > 0 and all(b is not None for b in ref_boxes)

        if has_cand_boxes and has_ref_boxes:
            # Full Hungarian bipartite matching via bounding box IoU with class affinity tie-breaking
            m = len(cand_list)
            k = len(ref_list)
            cost_matrix = np.zeros((m, k), dtype=np.float64)

            for i in range(m):
                for j in range(k):
                    b_a = cand_boxes[i]
                    b_b = ref_boxes[j]
                    iou = compute_bbox_iou(b_a, b_b) if (b_a is not None and b_b is not None) else 0.0
                    cand_cid = cand_list[i].get("class_id", cand_list[i].get("category_id"))
                    ref_cid = ref_list[j].get("class_id", ref_list[j].get("category_id"))
                    # Small class penalty (0.05) ensures spatial proximity dominates,
                    # but ties between overlapping/co-located boxes favor matching classes.
                    class_penalty = 0.05 if (cand_cid is not None and ref_cid is not None and cand_cid != ref_cid) else 0.0
                    cost_matrix[i, j] = (1.0 - iou) + class_penalty

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            assigned_cands = set(row_ind)
            assigned_refs = set(col_ind)

            for i, j in zip(row_ind, col_ind):
                # Calculate raw spatial IoU for discrepancy checks
                iou = compute_bbox_iou(cand_boxes[i], ref_boxes[j])
                cand_cid = cand_list[i].get("class_id", cand_list[i].get("category_id"))
                ref_cid = ref_list[j].get("class_id", ref_list[j].get("category_id"))

                if iou >= iou_match_threshold:
                    if cand_cid != ref_cid:
                        obj_class_changes += 1
                        img_has_anomaly = True
                        anomalies.append({
                            "file": fname,
                            "issue": "CLASS_CHANGE",
                            "candidate_class": cand_cid,
                            "reference_class": ref_cid,
                            "iou": round(float(iou), 4),
                            "candidate_bbox": cand_boxes[i],
                            "reference_bbox": ref_boxes[j],
                        })
                    elif iou < iou_clean_threshold:
                        obj_bbox_modifications += 1
                        img_has_anomaly = True
                        anomalies.append({
                            "file": fname,
                            "issue": "BBOX_MODIFIED",
                            "class_id": cand_cid,
                            "iou": round(float(iou), 4),
                            "candidate_bbox": cand_boxes[i],
                            "reference_bbox": ref_boxes[j],
                        })
                    else:
                        obj_matched_clean += 1
                else:
                    # IoU below match threshold: treat as separate missing + extra objects
                    obj_additional += 1
                    obj_missing += 1
                    img_has_anomaly = True
                    anomalies.append({
                        "file": fname,
                        "issue": "ADDITIONAL_ANNOTATION",
                        "candidate_class": cand_cid,
                        "candidate_bbox": cand_boxes[i],
                    })
                    anomalies.append({
                        "file": fname,
                        "issue": "MISSING_ANNOTATION",
                        "reference_class": ref_cid,
                        "reference_bbox": ref_boxes[j],
                    })

            # Unassigned candidates
            for i in range(m):
                if i not in assigned_cands:
                    obj_additional += 1
                    img_has_anomaly = True
                    anomalies.append({
                        "file": fname,
                        "issue": "ADDITIONAL_ANNOTATION",
                        "candidate_class": cand_list[i].get("class_id", cand_list[i].get("category_id")),
                        "candidate_bbox": cand_boxes[i],
                    })

            # Unassigned references
            for j in range(k):
                if j not in assigned_refs:
                    obj_missing += 1
                    img_has_anomaly = True
                    anomalies.append({
                        "file": fname,
                        "issue": "MISSING_ANNOTATION",
                        "reference_class": ref_list[j].get("class_id", ref_list[j].get("category_id")),
                        "reference_bbox": ref_boxes[j],
                    })

        elif len(cand_list) == 0 and len(ref_list) == 0:
            # Both empty - clean
            pass
        elif len(cand_list) == 0 and len(ref_list) > 0:
            obj_missing += len(ref_list)
            img_has_anomaly = True
            for r in ref_list:
                anomalies.append({
                    "file": fname,
                    "issue": "MISSING_ANNOTATION",
                    "reference_class": r.get("class_id", r.get("category_id")),
                })
        elif len(cand_list) > 0 and len(ref_list) == 0:
            obj_additional += len(cand_list)
            img_has_anomaly = True
            for c in cand_list:
                anomalies.append({
                    "file": fname,
                    "issue": "ADDITIONAL_ANNOTATION",
                    "candidate_class": c.get("class_id", c.get("category_id")),
                })
        else:
            # Classification-only labels without bounding boxes (multiset difference matching)
            cand_classes = [a.get("class_id", a.get("category_id")) for a in cand_list]
            ref_classes = [a.get("class_id", a.get("category_id")) for a in ref_list]
            
            c_cand = Counter(cand_classes)
            c_ref = Counter(ref_classes)
            common = c_cand & c_ref
            clean_count = sum(common.values())
            obj_matched_clean += clean_count

            diff_cand = list((c_cand - common).elements())
            diff_ref = list((c_ref - common).elements())

            # Pair up remaining as class changes where possible
            min_len = min(len(diff_cand), len(diff_ref))
            for i in range(min_len):
                obj_class_changes += 1
                img_has_anomaly = True
                anomalies.append({
                    "file": fname,
                    "issue": "CLASS_CHANGE",
                    "candidate_class": diff_cand[i],
                    "reference_class": diff_ref[i],
                })

            for cc in diff_cand[min_len:]:
                obj_additional += 1
                img_has_anomaly = True
                anomalies.append({
                    "file": fname,
                    "issue": "ADDITIONAL_ANNOTATION",
                    "candidate_class": cc,
                })

            for rc in diff_ref[min_len:]:
                obj_missing += 1
                img_has_anomaly = True
                anomalies.append({
                    "file": fname,
                    "issue": "MISSING_ANNOTATION",
                    "reference_class": rc,
                })

        if img_has_anomaly:
            img_anomaly_count += 1
        else:
            img_clean_count += 1

    total_ref_annotations = sum(len(reference_annotations[f]) for f in reference_annotations)
    flip_count = obj_class_changes + obj_missing + obj_additional + obj_bbox_modifications
    total_baseline = max(total_ref_annotations, 1)
    flip_pct = flip_count / total_baseline

    result = "PASS"
    severity = "INFO"
    if flip_count > 0 and flip_pct >= flip_threshold:
        result = "ANOMALY_DETECTED"
        severity = "CRITICAL" if flip_pct >= 0.10 else "HIGH"

    return {
        "check_id": "SEC-DS-002",
        "name": "Label Flipping & Integrity Detection",
        "result": result,
        "severity": severity,
        "confidence": "HIGH",  # deterministic comparison against reference
        "evidence": {
            "files_compared": len(common_files),
            "annotations_compared": total_ref_annotations,
            "flip_count": flip_count,
            "flip_fraction": round(flip_pct, 4),
            "threshold_fraction": flip_threshold,
            "object_level_breakdown": {
                "class_changes": obj_class_changes,
                "bbox_modifications": obj_bbox_modifications,
                "missing_objects": obj_missing,
                "additional_objects": obj_additional,
                "matched_clean": obj_matched_clean,
            },
            "image_level_breakdown": {
                "images_clean": img_clean_count,
                "images_with_anomalies": img_anomaly_count,
                "missing_files": len(missing_files),
                "extra_files": len(extra_files),
            },
            "examples": anomalies[:25],
        },
        "description": (
            f"{flip_count} annotation anomalies detected across {len(common_files)} compared files "
            f"({obj_class_changes} class flips, {obj_bbox_modifications} bbox modifications, "
            f"{obj_missing} missing objects, {obj_additional} additional objects)."
            if flip_count > 0 else
            f"All {total_ref_annotations} annotation(s) across {len(common_files)} files match reference perfectly."
        ),
        "recommended_action": (
            "Review annotation mismatches using Hungarian IoU matches. "
            "Re-annotate modified bounding boxes and quarantine flipped labels."
            if flip_count > 0 else "No action required."
        ),
        "limitation": (
            "Requires a trusted reference annotation set. "
            "Hungarian bipartite matching uses IoU threshold of 0.5 for spatial association."
        ),
    }


# ---------------------------------------------------------------------------
# 3. Trigger pattern insertion (patch-based low-entropy search)
# ---------------------------------------------------------------------------

def check_trigger_patterns(
    image_paths: List[Path],
    patch_size: int = 16,
    entropy_threshold: float = 1.0,
    min_flagged_patches: int = 1,
    audit_id: str = "",
) -> Dict[str, Any]:
    """
    Search for low-entropy (uniform / near-uniform) rectangular patches
    that may indicate a backdoor trigger inserted into images.

    Limitation: this is a heuristic. Sophisticated triggers with texture
    may have higher entropy and evade detection. Low entropy is a necessary
    but not sufficient condition for trigger insertion.

    Args:
        image_paths: list of image file paths to scan
        patch_size: side length of sliding window (pixels)
        entropy_threshold: patches below this Shannon entropy are flagged
        min_flagged_patches: minimum patch flags per image to report that image
    """
    flagged_images: List[Dict] = []
    error_images: List[str] = []
    continuous_scores: Dict[str, float] = {}

    for img_path in image_paths:
        arr = _load_rgb(img_path)
        if arr is None:
            error_images.append(str(img_path))
            continue

        h, w, _ = arr.shape
        flagged_patches: List[Dict] = []
        min_ent = 8.0  # theoretical max for 8-bit uniform is 8.0

        for y in range(0, h - patch_size + 1, patch_size):
            for x in range(0, w - patch_size + 1, patch_size):
                patch = arr[y:y + patch_size, x:x + patch_size]
                ent = _patch_entropy(patch)
                if ent < min_ent:
                    min_ent = ent
                
                # Only log heavily anomalous patches to prevent flooding SQLite
                if ent < entropy_threshold + 0.5:
                    if audit_id and log_event:
                        res_str = "TRUE" if ent < entropy_threshold else "FALSE"
                        stat_str = "WARNING" if ent < entropy_threshold else "PASS"
                        log_event(audit_id, "Trigger Analysis", "Patch Shannon Entropy", 
                                  {"file": img_path.name, "x": x, "y": y, "w": patch_size}, 
                                  "H(X) = -sum(p * log2(p))", 
                                  {"H": ent, "threshold": entropy_threshold}, 
                                  res_str, f"H < {entropy_threshold:.2f}", stat_str, "", 0, stat_str)

                if ent < entropy_threshold:
                    flagged_patches.append({
                        "top_left": [x, y],
                        "size": patch_size,
                        "entropy": round(ent, 4),
                    })

        continuous_scores[img_path.name] = round(float(min_ent), 4)

        if len(flagged_patches) >= min_flagged_patches:
            flagged_images.append({
                "filename": img_path.name,
                "flagged_patch_count": len(flagged_patches),
                "min_patch_entropy": round(float(min_ent), 4),
                "patches": flagged_patches[:5],  # cap evidence output
            })

    result = "PASS"
    severity = "INFO"
    if flagged_images:
        result = "ANOMALY_DETECTED"
        severity = "MEDIUM"

    return {
        "check_id": "SEC-DS-003",
        "name": "Trigger Pattern Insertion Detection",
        "result": result,
        "severity": severity,
        "confidence": "LOW",  # heuristic; cannot confirm intentional trigger
        "evidence": {
            "images_scanned": len(image_paths),
            "flagged_images": flagged_images,
            "error_images": error_images,
            "patch_size": patch_size,
            "entropy_threshold": entropy_threshold,
            "continuous_scores": continuous_scores,
        },
        "description": (
            f"{len(flagged_images)} image(s) contain low-entropy patches "
            f"(Shannon entropy < {entropy_threshold}). "
            "Such patches may indicate a backdoor trigger insertion."
        ),
        "recommended_action": (
            "Visually inspect flagged images. "
            "Cross-reference with annotation provenance. "
            "If trigger candidates are confirmed, quarantine and re-collect from original source."
        ),
        "limitation": (
            "Entropy-based trigger detection is a heuristic with low confidence. "
            "Natural low-texture regions (sky, walls) will also trigger this flag. "
            "Sophisticated texture-based triggers may not be detected. "
            "A positive result indicates an anomaly requiring review, NOT a confirmed attack."
        ),
        "trace": {
            "operation_id": "TRIGGER_ENTROPY",
            "operation_type": "Shannon Entropy Low-Texture Heuristic",
            "formula": "H(X) = -sum(p_i * log2(p_i))",
            "inputs": {
                "images_scanned": len(image_paths),
                "patch_size": patch_size
            },
            "intermediate_values": {
                "flagged_patches_found": sum(img["flagged_patch_count"] for img in flagged_images) if flagged_images else 0
            },
            "result": f"{len(flagged_images)} flagged",
            "threshold": entropy_threshold,
            "comparison": f"H(X) < {entropy_threshold}",
            "decision": result,
            "explanation": "Calculates Shannon entropy of pixel value distribution within sliding windows. A low entropy indicates a highly uniform patch which acts as a heuristic indicator for potential backdoor triggers, though benign uniform surfaces will also flag."
        }
    }


# ---------------------------------------------------------------------------
# 4. Dataset poisoning (global pixel distribution outlier)
# ---------------------------------------------------------------------------

def check_data_poisoning(
    records: List[Dict[str, Any]],
    z_threshold: float = 3.0,
    reference_records: Optional[List[Dict[str, Any]]] = None,
    use_robust_mad: bool = False,
    audit_id: str = "",
) -> Dict[str, Any]:
    """
    Detect potential poisoned images via statistical outlier analysis
    on pixel mean / standard deviation vectors.

    Supports:
      - Standard parametric Z-scores (mean / standard deviation)
      - Robust non-parametric Modified Z-scores (Median / MAD)
      - Reference-calibrated distribution scoring (when reference_records provided)

    Args:
        records: per-image records containing 'pixel_stats'
        z_threshold: number of standard deviations / MADs to flag as outlier
        reference_records: optional trusted reference records for baseline calibration
        use_robust_mad: whether to use Median/MAD instead of mean/std
    """
    def _extract_vecs(recs: List[Dict[str, Any]]) -> Tuple[List[str], np.ndarray]:
        names, vecs = [], []
        for r in recs:
            ps = r.get("pixel_stats")
            if isinstance(ps, dict):
                v = [ps.get("mean_r", 0.0), ps.get("mean_g", 0.0), ps.get("mean_b", 0.0),
                     ps.get("std_r", 0.0),  ps.get("std_g", 0.0),  ps.get("std_b", 0.0)]
                names.append(r.get("filename", ""))
                vecs.append(v)
        return names, np.array(vecs, dtype=np.float64) if vecs else np.empty((0, 6))

    cand_names, cand_vecs = _extract_vecs(records)

    if len(cand_vecs) < 3 and reference_records is None:
        return {
            "check_id": "SEC-DS-004",
            "name": "Data Poisoning (Statistical Outlier)",
            "result": "NOT ASSESSED",
            "severity": "INFO",
            "confidence": "NOT ASSESSED",
            "evidence": {"reason": "Insufficient samples (< 3) for statistical analysis"},
            "description": "Cannot perform statistical outlier analysis with fewer than 3 images.",
            "recommended_action": "Provide a larger dataset sample for meaningful analysis.",
            "limitation": "Z-score analysis requires sufficient sample size.",
        }

    # Reference-calibrated vs self-referenced
    if reference_records and len(reference_records) >= 3:
        _, ref_vecs = _extract_vecs(reference_records)
        baseline_vecs = ref_vecs if len(ref_vecs) >= 3 else cand_vecs
    else:
        baseline_vecs = cand_vecs

    if use_robust_mad:
        loc = np.median(baseline_vecs, axis=0)
        mad = np.median(np.abs(baseline_vecs - loc), axis=0)
        scale = 1.4826 * mad
        # If MAD is near-zero (e.g., >50% identical samples in reference),
        # fall back to Mean Absolute Deviation or sample std to prevent denominator blow-up.
        zero_mad = scale < 1e-6
        if np.any(zero_mad):
            mean_ad = 1.2533 * np.mean(np.abs(baseline_vecs - loc), axis=0)
            sample_std = baseline_vecs.std(axis=0)
            fallback = np.where(mean_ad >= 1e-6, mean_ad, sample_std)
            # If feature has zero dispersion, set scale to 1.0 (1 unit intensity) to prevent division by zero
            fallback = np.where(fallback < 1e-6, 1.0, fallback)
            scale = np.where(zero_mad, fallback, scale)
    else:
        loc = baseline_vecs.mean(axis=0)
        scale = baseline_vecs.std(axis=0)
        scale = np.where(scale < 1e-6, 1.0, scale)

    z_scores = np.abs((cand_vecs - loc) / scale)
    max_z_per_sample = np.max(z_scores, axis=1) if len(z_scores) > 0 else np.array([])

    flagged: List[Dict] = []
    continuous_scores: Dict[str, float] = {}

    for i, (name, z_vec, max_z) in enumerate(zip(cand_names, z_scores, max_z_per_sample)):
        continuous_scores[name] = round(float(max_z), 4)
        if max_z > z_threshold:
            flagged.append({
                "filename": name,
                "max_z_score": round(float(max_z), 4),
                "channel_z_scores": {
                    "mean_r": round(float(z_vec[0]), 4),
                    "mean_g": round(float(z_vec[1]), 4),
                    "mean_b": round(float(z_vec[2]), 4),
                    "std_r":  round(float(z_vec[3]), 4),
                    "std_g":  round(float(z_vec[4]), 4),
                    "std_b":  round(float(z_vec[5]), 4),
                },
            })

    result = "PASS"
    severity = "INFO"
    if flagged:
        result = "STATISTICAL_SHIFT"
        severity = "MEDIUM"

    return {
        "check_id": "SEC-DS-004",
        "name": "Distribution Shift (Statistical Outlier)",
        "result": result,
        "severity": severity,
        "confidence": "LOW",
        "evidence": {
            "images_analysed": len(cand_vecs),
            "z_threshold": z_threshold,
            "method": "robust_mad" if use_robust_mad else "standard_zscore",
            "calibrated_against_reference": reference_records is not None,
            "flagged_images": flagged,
            "continuous_scores": continuous_scores,
        },
        "description": (
            f"{len(flagged)} image(s) have pixel statistics exceeding {z_threshold} "
            f"{'MAD' if use_robust_mad else 'standard deviations'} from the baseline distribution."
            if flagged else
            f"Zero statistical outliers detected across {len(cand_vecs)} images."
        ),
        "recommended_action": (
            "Review flagged images for unusual content, overlaid objects, or synthetic artifacts."
            if flagged else "No action required."
        ),
        "limitation": (
            "Z-score outlier detection is a statistical indicator, not definitive proof of attack. "
            "Self-referenced calculation can be vulnerable to poison masking."
        ),
    }


# ---------------------------------------------------------------------------
# 5. Out-of-Distribution (OOD) sample detection via Mahalanobis distance
# ---------------------------------------------------------------------------

def check_ood_samples(
    records: List[Dict[str, Any]],
    reference_records: Optional[List[Dict[str, Any]]] = None,
    threshold: Optional[float] = None,
    regularization: float = 1e-4,
    audit_id: str = "",
) -> Dict[str, Any]:
    """
    Detect Out-of-Distribution (OOD) samples using Mahalanobis distance across
    pixel color and distribution features.

    Supports:
      - Reference-calibrated distribution modeling (prevents self-contamination)
      - Covariance ridge shrinkage regularization (lambda * I) for numerical stability
      - Pre-frozen calibration thresholds
      - Continuous anomaly distance emission for AUROC/AUPR evaluation

    Args:
        records: per-image records containing 'pixel_stats'
        reference_records: optional reference records to fit distribution mean and covariance
        threshold: pre-frozen threshold (if None, computes mean + 2.5 * std from baseline)
        regularization: diagonal shrinkage added to covariance matrix (default 1e-4)
    """
    def _extract_means(recs: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        valid_r, vecs = [], []
        for r in recs:
            ps = r.get("pixel_stats")
            if isinstance(ps, dict) and "mean_r" in ps and "mean_g" in ps and "mean_b" in ps:
                vecs.append([float(ps["mean_r"]), float(ps["mean_g"]), float(ps["mean_b"])])
                valid_r.append(r)
        return valid_r, np.array(vecs, dtype=np.float64) if vecs else np.empty((0, 3))

    valid_cand_records, cand_arr = _extract_means(records)

    if len(cand_arr) < 3 and reference_records is None:
        return {
            "check_id": "SEC-DS-005",
            "name": "Out-of-Distribution (OOD) Detection",
            "result": "NOT_ASSESSED",
            "severity": "INFO",
            "confidence": "NOT_ASSESSED",
            "evidence": {"reason": "Insufficient samples (< 3) for Mahalanobis covariance estimation."},
            "description": "Cannot perform OOD covariance estimation with fewer than 3 samples.",
            "recommended_action": "Provide a larger dataset for distribution analysis.",
            "limitation": "Mahalanobis distance requires sufficient sample size to estimate covariance.",
        }

    # Use reference records to establish ground truth distribution if available
    if reference_records and len(reference_records) >= 3:
        _, ref_arr = _extract_means(reference_records)
        baseline_arr = ref_arr if len(ref_arr) >= 3 else cand_arr
    else:
        baseline_arr = cand_arr

    global_mean = baseline_arr.mean(axis=0)
    cov = np.cov(baseline_arr.T)
    # Apply diagonal ridge shrinkage to ensure positive-definiteness and numerical stability
    if cov.ndim == 2:
        cov_reg = cov + regularization * np.eye(cov.shape[0])
    else:
        cov_reg = cov + regularization

    try:
        inv_cov = np.linalg.pinv(cov_reg)
        dists = []
        continuous_scores: Dict[str, float] = {}

        for rec, vec in zip(valid_cand_records, cand_arr):
            diff = vec - global_mean
            sq_dist = diff @ inv_cov @ diff
            d = float(math.sqrt(max(0.0, sq_dist)))
            dists.append(d)
            continuous_scores[rec.get("filename", "")] = round(d, 4)

        if threshold is None:
            threshold = float(np.mean(dists) + 2.5 * np.std(dists)) if len(dists) > 1 else 1e9

        flagged: List[Dict[str, Any]] = []
        for rec, d in zip(valid_cand_records, dists):
            if d > threshold:
                flagged.append({
                    "filename": rec.get("filename"),
                    "mahalanobis_distance": round(d, 4),
                    "threshold": round(threshold, 4),
                })

        result = "PASS"
        severity = "INFO"
        if flagged:
            result = "ANOMALY_DETECTED"
            severity = "MEDIUM"

        return {
            "check_id": "SEC-DS-005",
            "name": "Out-of-Distribution (OOD) Detection",
            "result": result,
            "severity": severity,
            "confidence": "MEDIUM",
            "evidence": {
                "images_analyzed": len(valid_cand_records),
                "threshold": round(threshold, 4),
                "calibrated_against_reference": reference_records is not None,
                "flagged_candidates": flagged,
                "continuous_scores": continuous_scores,
            },
            "description": (
                f"{len(flagged)} image(s) deviate significantly from the baseline distribution "
                f"(Mahalanobis distance > {threshold:.2f})."
                if flagged else
                f"No OOD anomalies detected across {len(valid_cand_records)} images."
            ),
            "recommended_action": (
                "Review flagged candidate images for anomalous lighting, scene shifts, or synthetic artifacts."
                if flagged else "No action required."
            ),
            "limitation": (
                "Mahalanobis distance on channel distributions captures color balance and brightness. "
                "Semantic OOD shifts with matched color profiles may evade detection."
            ),
        }
    except Exception as exc:
        return {
            "check_id": "SEC-DS-005",
            "name": "Out-of-Distribution (OOD) Detection",
            "result": "NOT_ASSESSED",
            "severity": "INFO",
            "confidence": "NOT_ASSESSED",
            "evidence": {"error": str(exc)},
            "description": f"Mahalanobis distance computation failed: {exc}",
            "recommended_action": "Verify numeric stability of dataset pixel distributions.",
            "limitation": "Numerical singularity or matrix inversion failure.",
        }
