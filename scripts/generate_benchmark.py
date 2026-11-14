"""
scripts/generate_benchmark.py

Deterministic Controlled Benchmark Generator for Computer Vision Dataset Assurance.
Creates strictly partitioned datasets with zero experimental leakage:
  1. data/benchmark/reference/   - Trusted reference dataset (baseline distribution)
  2. data/benchmark/calibration/ - Threshold calibration split (used ONLY to freeze thresholds)
  3. data/benchmark/evaluation/  - Independent evaluation benchmark with explicit ground truth

Threat Scenarios Generated:
  - Clean: normal, multi-resolution, compression variations, lighting variations
  - Duplicate: exact byte copies, JPEG recompressed, resized, cropped, color-shifted
  - Annotation: class flips, missing labels, extra labels, bbox modifications, permuted labels
  - Trigger: solid corner squares, multi-size (10x10, 15x15, 25x25), multi-location, patterned
  - Statistical: brightness shifts, contrast shifts, color tinting, additive noise
  - OOD: cross-domain patterns, abstract color fields, noise distributions, semantic outliers
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
from PIL import Image, ImageEnhance, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = ROOT / "data" / "benchmark"
CLASS_NAMES = ["vehicle", "pedestrian", "traffic_sign", "cyclist"]

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def make_synthetic_image(rng: random.Random, width: int = 100, height: int = 100,
                         base_color: Tuple[int, int, int] = None, noise_sigma: float = 25.0) -> Image.Image:
    """Generate high-entropy synthetic base image with structured shapes."""
    if base_color is None:
        base_color = (rng.randint(60, 190), rng.randint(60, 190), rng.randint(60, 190))
    
    # Base background with noise
    arr = np.zeros((height, width, 3), dtype=np.float32)
    for c in range(3):
        arr[:, :, c] = base_color[c]
    noise = np.random.default_rng(rng.randint(0, 100000)).normal(0, noise_sigma, (height, width, 3))
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, "RGB")
    draw = ImageDraw.Draw(img)

    # Add 2-3 geometric object shapes simulating objects
    num_shapes = rng.randint(2, 4)
    for _ in range(num_shapes):
        shape_type = rng.choice(["rect", "circle"])
        x1 = rng.randint(10, width - 40)
        y1 = rng.randint(10, height - 40)
        x2 = x1 + rng.randint(15, 30)
        y2 = y1 + rng.randint(15, 30)
        col = (rng.randint(20, 240), rng.randint(20, 240), rng.randint(20, 240))
        if shape_type == "rect":
            draw.rectangle([x1, y1, x2, y2], fill=col, outline=(0, 0, 0))
        else:
            draw.ellipse([x1, y1, x2, y2], fill=col, outline=(0, 0, 0))

    return img

def save_yolo_sample(img: Image.Image, annotations: List[Dict[str, Any]],
                     images_dir: Path, labels_dir: Path, stem: str):
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    img_path = images_dir / f"{stem}.png"
    img.save(img_path)

    lbl_path = labels_dir / f"{stem}.txt"
    lines = []
    for a in annotations:
        cid = a["class_id"]
        xc = a["x_center"]
        yc = a["y_center"]
        w = a["width"]
        h = a["height"]
        lines.append(f"{cid} {xc:.4f} {yc:.4f} {w:.4f} {h:.4f}")
    lbl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

# ---------------------------------------------------------------------------
# 1. Reference Dataset (Clean Baseline)
# ---------------------------------------------------------------------------

def generate_reference_split(seed: int = 42, count: int = 30) -> Dict[str, Any]:
    rng = random.Random(seed)
    ref_dir = BENCHMARK_DIR / "reference"
    imgs_dir = ref_dir / "images"
    lbls_dir = ref_dir / "labels"
    imgs_dir.mkdir(parents=True, exist_ok=True)
    lbls_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}
    ground_truth = {}

    for i in range(count):
        stem = f"ref_{i:04d}"
        img = make_synthetic_image(rng, 100, 100, base_color=(120, 130, 140))
        # 1-2 bounding box annotations per image
        num_objects = rng.randint(1, 2)
        anns = []
        for j in range(num_objects):
            cid = (i + j) % len(CLASS_NAMES)
            xc = round(0.25 + 0.35 * j + rng.uniform(-0.05, 0.05), 4)
            yc = round(0.30 + 0.30 * j + rng.uniform(-0.05, 0.05), 4)
            w = round(0.20 + rng.uniform(-0.02, 0.04), 4)
            h = round(0.20 + rng.uniform(-0.02, 0.04), 4)
            anns.append({"class_id": cid, "x_center": xc, "y_center": yc, "width": w, "height": h})
        
        save_yolo_sample(img, anns, imgs_dir, lbls_dir, stem)
        img_p = imgs_dir / f"{stem}.png"
        manifest[f"{stem}.png"] = sha256_file(img_p)
        ground_truth[f"{stem}.png"] = {
            "is_anomaly": False,
            "threat_category": "clean",
            "scenario": "reference_baseline",
            "annotations": anns
        }

    (ref_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (ref_dir / "ground_truth.json").write_text(json.dumps(ground_truth, indent=2), encoding="utf-8")
    print(f"[OK] Reference dataset generated: {count} samples in {ref_dir}")
    return ground_truth

# ---------------------------------------------------------------------------
# 2. Calibration Dataset (Threshold Tuning Only)
# ---------------------------------------------------------------------------

def generate_calibration_split(seed: int = 1042) -> Dict[str, Any]:
    rng = random.Random(seed)
    cal_dir = BENCHMARK_DIR / "calibration"
    imgs_dir = cal_dir / "images"
    lbls_dir = cal_dir / "labels"
    imgs_dir.mkdir(parents=True, exist_ok=True)
    lbls_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}
    ground_truth = {}

    # 15 clean calibration samples
    clean_stems = []
    for i in range(15):
        stem = f"cal_clean_{i:04d}"
        clean_stems.append(stem)
        img = make_synthetic_image(rng, 100, 100, base_color=(120, 130, 140))
        anns = [{"class_id": i % 4, "x_center": 0.4, "y_center": 0.4, "width": 0.25, "height": 0.25}]
        save_yolo_sample(img, anns, imgs_dir, lbls_dir, stem)
        img_p = imgs_dir / f"{stem}.png"
        manifest[f"{stem}.png"] = sha256_file(img_p)
        ground_truth[f"{stem}.png"] = {"is_anomaly": False, "threat_category": "clean", "scenario": "clean"}

    # 2 exact duplicates
    for i in range(2):
        stem = f"cal_dup_{i:04d}"
        src_stem = clean_stems[i]
        src_img = Image.open(imgs_dir / f"{src_stem}.png")
        anns = [{"class_id": i % 4, "x_center": 0.4, "y_center": 0.4, "width": 0.25, "height": 0.25}]
        save_yolo_sample(src_img, anns, imgs_dir, lbls_dir, stem)
        manifest[f"{stem}.png"] = sha256_file(imgs_dir / f"{stem}.png")
        ground_truth[f"{stem}.png"] = {"is_anomaly": True, "threat_category": "duplicate", "scenario": "exact_duplicate"}

    # 2 label flips
    for i in range(2):
        stem = f"cal_labelflip_{i:04d}"
        img = make_synthetic_image(rng, 100, 100)
        # Flip class from 0 to 3
        anns = [{"class_id": 3, "x_center": 0.4, "y_center": 0.4, "width": 0.25, "height": 0.25}]
        save_yolo_sample(img, anns, imgs_dir, lbls_dir, stem)
        manifest[f"{stem}.png"] = sha256_file(imgs_dir / f"{stem}.png")
        ground_truth[f"{stem}.png"] = {
            "is_anomaly": True,
            "threat_category": "annotation",
            "scenario": "label_flip",
            "reference_annotations": [{"class_id": 0, "x_center": 0.4, "y_center": 0.4, "width": 0.25, "height": 0.25}]
        }

    # 2 trigger patches (20x20 solid red in bottom-right)
    for i in range(2):
        stem = f"cal_trigger_{i:04d}"
        img = make_synthetic_image(rng, 100, 100)
        draw = ImageDraw.Draw(img)
        draw.rectangle([75, 75, 95, 95], fill=(255, 0, 0))
        anns = [{"class_id": 1, "x_center": 0.4, "y_center": 0.4, "width": 0.25, "height": 0.25}]
        save_yolo_sample(img, anns, imgs_dir, lbls_dir, stem)
        manifest[f"{stem}.png"] = sha256_file(imgs_dir / f"{stem}.png")
        ground_truth[f"{stem}.png"] = {"is_anomaly": True, "threat_category": "trigger", "scenario": "solid_trigger"}

    # 2 statistical shifts (extreme brightness)
    for i in range(2):
        stem = f"cal_stat_{i:04d}"
        img = make_synthetic_image(rng, 100, 100, base_color=(230, 240, 240))
        anns = [{"class_id": 2, "x_center": 0.4, "y_center": 0.4, "width": 0.25, "height": 0.25}]
        save_yolo_sample(img, anns, imgs_dir, lbls_dir, stem)
        manifest[f"{stem}.png"] = sha256_file(imgs_dir / f"{stem}.png")
        ground_truth[f"{stem}.png"] = {"is_anomaly": True, "threat_category": "statistical", "scenario": "brightness_outlier"}

    # 2 OOD samples (grayscale color distribution)
    for i in range(2):
        stem = f"cal_ood_{i:04d}"
        arr = np.full((100, 100, 3), 200, dtype=np.uint8)
        img = Image.fromarray(arr, "RGB")
        anns = [{"class_id": 0, "x_center": 0.5, "y_center": 0.5, "width": 0.3, "height": 0.3}]
        save_yolo_sample(img, anns, imgs_dir, lbls_dir, stem)
        manifest[f"{stem}.png"] = sha256_file(imgs_dir / f"{stem}.png")
        ground_truth[f"{stem}.png"] = {"is_anomaly": True, "threat_category": "ood", "scenario": "distribution_shift"}

    (cal_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (cal_dir / "ground_truth.json").write_text(json.dumps(ground_truth, indent=2), encoding="utf-8")
    print(f"[OK] Calibration dataset generated: {len(ground_truth)} samples in {cal_dir}")
    return ground_truth

# ---------------------------------------------------------------------------
# 3. Evaluation Dataset (Independent Test Benchmark)
# ---------------------------------------------------------------------------

def generate_evaluation_split(seed: int = 2042) -> Dict[str, Any]:
    rng = random.Random(seed)
    eval_dir = BENCHMARK_DIR / "evaluation"
    imgs_dir = eval_dir / "images"
    lbls_dir = eval_dir / "labels"
    imgs_dir.mkdir(parents=True, exist_ok=True)
    lbls_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}
    ground_truth = {}
    reference_annotations_map = {}

    clean_source_stems = []

    # A. Clean Subsets (30 clean images)
    # 1. Standard normal images (15)
    for i in range(15):
        stem = f"eval_clean_std_{i:04d}"
        clean_source_stems.append(stem)
        img = make_synthetic_image(rng, 100, 100, base_color=(120, 130, 140))
        anns = [
            {"class_id": i % 4, "x_center": 0.35, "y_center": 0.35, "width": 0.25, "height": 0.25},
            {"class_id": (i + 1) % 4, "x_center": 0.70, "y_center": 0.70, "width": 0.20, "height": 0.20}
        ]
        save_yolo_sample(img, anns, imgs_dir, lbls_dir, stem)
        fn = f"{stem}.png"
        manifest[fn] = sha256_file(imgs_dir / fn)
        ground_truth[fn] = {"is_anomaly": False, "threat_category": "clean", "scenario": "standard_clean", "clean_type": "standard"}
        reference_annotations_map[fn] = anns

    # 2. Multi-resolution clean images (5)
    resolutions = [(50, 50), (80, 80), (120, 120), (150, 150), (200, 200)]
    for i, (w, h) in enumerate(resolutions):
        stem = f"eval_clean_res_{i:04d}_{w}x{h}"
        img = make_synthetic_image(rng, w, h, base_color=(120, 130, 140))
        anns = [{"class_id": i % 4, "x_center": 0.5, "y_center": 0.5, "width": 0.3, "height": 0.3}]
        save_yolo_sample(img, anns, imgs_dir, lbls_dir, stem)
        fn = f"{stem}.png"
        manifest[fn] = sha256_file(imgs_dir / fn)
        ground_truth[fn] = {"is_anomaly": False, "threat_category": "clean", "scenario": "multi_resolution", "resolution": f"{w}x{h}"}
        reference_annotations_map[fn] = anns

    # 3. Compression variations clean (5)
    for i in range(5):
        stem = f"eval_clean_comp_{i:04d}"
        img = make_synthetic_image(rng, 100, 100, base_color=(120, 130, 140))
        # Save as JPEG with varied quality then reload
        q = [50, 65, 75, 85, 95][i]
        tmp_jpg = eval_dir / f"temp_{stem}.jpg"
        img.save(tmp_jpg, "JPEG", quality=q)
        img_reloaded = Image.open(tmp_jpg).convert("RGB")
        tmp_jpg.unlink()
        anns = [{"class_id": i % 4, "x_center": 0.4, "y_center": 0.4, "width": 0.25, "height": 0.25}]
        save_yolo_sample(img_reloaded, anns, imgs_dir, lbls_dir, stem)
        fn = f"{stem}.png"
        manifest[fn] = sha256_file(imgs_dir / fn)
        ground_truth[fn] = {"is_anomaly": False, "threat_category": "clean", "scenario": "compression_variation", "quality": q}
        reference_annotations_map[fn] = anns

    # 4. Lighting variations clean (5)
    for i in range(5):
        stem = f"eval_clean_light_{i:04d}"
        img = make_synthetic_image(rng, 100, 100, base_color=(120, 130, 140))
        factor = [0.85, 0.90, 1.05, 1.15, 1.20][i]
        enhancer = ImageEnhance.Brightness(img)
        img_adj = enhancer.enhance(factor)
        anns = [{"class_id": i % 4, "x_center": 0.4, "y_center": 0.4, "width": 0.25, "height": 0.25}]
        save_yolo_sample(img_adj, anns, imgs_dir, lbls_dir, stem)
        fn = f"{stem}.png"
        manifest[fn] = sha256_file(imgs_dir / fn)
        ground_truth[fn] = {"is_anomaly": False, "threat_category": "clean", "scenario": "lighting_variation", "brightness_factor": factor}
        reference_annotations_map[fn] = anns

    # B. Duplicate Scenarios (20 duplicate images)
    # 1. Exact bitwise duplicates (5)
    for i in range(5):
        stem = f"eval_dup_exact_{i:04d}"
        src_stem = clean_source_stems[i]
        src_img = Image.open(imgs_dir / f"{src_stem}.png")
        anns = list(reference_annotations_map[f"{src_stem}.png"])
        save_yolo_sample(src_img, anns, imgs_dir, lbls_dir, stem)
        fn = f"{stem}.png"
        manifest[fn] = sha256_file(imgs_dir / fn)
        ground_truth[fn] = {
            "is_anomaly": True,
            "threat_category": "duplicate",
            "scenario": "exact_duplicate",
            "source_file": f"{src_stem}.png"
        }
        reference_annotations_map[fn] = anns

    # 2. Recompressed near-duplicates (JPEG Q60/Q70/Q80) (5)
    for i in range(5):
        stem = f"eval_dup_recomp_{i:04d}"
        src_stem = clean_source_stems[5 + i]
        src_img = Image.open(imgs_dir / f"{src_stem}.png")
        q = [60, 65, 70, 75, 80][i]
        tmp_j = eval_dir / f"tmp_{stem}.jpg"
        src_img.save(tmp_j, "JPEG", quality=q)
        recomp_img = Image.open(tmp_j).convert("RGB")
        tmp_j.unlink()
        anns = list(reference_annotations_map[f"{src_stem}.png"])
        save_yolo_sample(recomp_img, anns, imgs_dir, lbls_dir, stem)
        fn = f"{stem}.png"
        manifest[fn] = sha256_file(imgs_dir / fn)
        ground_truth[fn] = {
            "is_anomaly": True,
            "threat_category": "duplicate",
            "scenario": "recompressed_near_duplicate",
            "quality": q,
            "source_file": f"{src_stem}.png"
        }
        reference_annotations_map[fn] = anns

    # 3. Resized near-duplicates (3)
    for i, target_sz in enumerate([(90, 90), (105, 105), (110, 110)]):
        stem = f"eval_dup_resize_{i:04d}"
        src_stem = clean_source_stems[10 + i]
        src_img = Image.open(imgs_dir / f"{src_stem}.png")
        # Resize then resize back to 100x100
        resized = src_img.resize(target_sz, Image.Resampling.BILINEAR).resize((100, 100), Image.Resampling.BILINEAR)
        anns = list(reference_annotations_map[f"{src_stem}.png"])
        save_yolo_sample(resized, anns, imgs_dir, lbls_dir, stem)
        fn = f"{stem}.png"
        manifest[fn] = sha256_file(imgs_dir / fn)
        ground_truth[fn] = {
            "is_anomaly": True,
            "threat_category": "duplicate",
            "scenario": "resized_near_duplicate",
            "source_file": f"{src_stem}.png"
        }
        reference_annotations_map[fn] = anns

    # 4. Cropped near-duplicates (3)
    for i in range(3):
        stem = f"eval_dup_crop_{i:04d}"
        src_stem = clean_source_stems[12 + i]
        src_img = Image.open(imgs_dir / f"{src_stem}.png")
        w, h = src_img.size
        # 5% margin crop
        cropped = src_img.crop((3, 3, w - 3, h - 3)).resize((w, h), Image.Resampling.BILINEAR)
        anns = list(reference_annotations_map[f"{src_stem}.png"])
        save_yolo_sample(cropped, anns, imgs_dir, lbls_dir, stem)
        fn = f"{stem}.png"
        manifest[fn] = sha256_file(imgs_dir / fn)
        ground_truth[fn] = {
            "is_anomaly": True,
            "threat_category": "duplicate",
            "scenario": "cropped_near_duplicate",
            "source_file": f"{src_stem}.png"
        }
        reference_annotations_map[fn] = anns

    # 5. Color shift near-duplicates (4)
    for i in range(4):
        stem = f"eval_dup_colorshift_{i:04d}"
        src_stem = clean_source_stems[i]
        src_img = Image.open(imgs_dir / f"{src_stem}.png")
        arr = np.array(src_img, dtype=np.int16)
        # Shift green channel by +10
        arr[:, :, 1] = np.clip(arr[:, :, 1] + 12, 0, 255)
        shifted_img = Image.fromarray(arr.astype(np.uint8))
        anns = list(reference_annotations_map[f"{src_stem}.png"])
        save_yolo_sample(shifted_img, anns, imgs_dir, lbls_dir, stem)
        fn = f"{stem}.png"
        manifest[fn] = sha256_file(imgs_dir / fn)
        ground_truth[fn] = {
            "is_anomaly": True,
            "threat_category": "duplicate",
            "scenario": "colorshift_near_duplicate",
            "source_file": f"{src_stem}.png"
        }
        reference_annotations_map[fn] = anns

    # C. Annotation Scenarios (25 images)
    # 1. Class flips (5)
    for i in range(5):
        stem = f"eval_ann_flip_{i:04d}"
        img = make_synthetic_image(rng, 100, 100)
        ref_anns = [{"class_id": 0, "x_center": 0.4, "y_center": 0.4, "width": 0.3, "height": 0.3}]
        # Flip class_id to 2
        cand_anns = [{"class_id": 2, "x_center": 0.4, "y_center": 0.4, "width": 0.3, "height": 0.3}]
        save_yolo_sample(img, cand_anns, imgs_dir, lbls_dir, stem)
        fn = f"{stem}.png"
        manifest[fn] = sha256_file(imgs_dir / fn)
        ground_truth[fn] = {
            "is_anomaly": True,
            "threat_category": "annotation",
            "scenario": "class_flip",
            "expected_issue": "CLASS_CHANGE",
            "reference_annotations": ref_anns
        }
        reference_annotations_map[fn] = ref_anns

    # 2. Missing annotations (5)
    for i in range(5):
        stem = f"eval_ann_missing_{i:04d}"
        img = make_synthetic_image(rng, 100, 100)
        ref_anns = [
            {"class_id": 1, "x_center": 0.3, "y_center": 0.3, "width": 0.2, "height": 0.2},
            {"class_id": 2, "x_center": 0.7, "y_center": 0.7, "width": 0.2, "height": 0.2}
        ]
        # Candidate omits the second object
        cand_anns = [{"class_id": 1, "x_center": 0.3, "y_center": 0.3, "width": 0.2, "height": 0.2}]
        save_yolo_sample(img, cand_anns, imgs_dir, lbls_dir, stem)
        fn = f"{stem}.png"
        manifest[fn] = sha256_file(imgs_dir / fn)
        ground_truth[fn] = {
            "is_anomaly": True,
            "threat_category": "annotation",
            "scenario": "missing_annotation",
            "expected_issue": "MISSING_ANNOTATION",
            "reference_annotations": ref_anns
        }
        reference_annotations_map[fn] = ref_anns

    # 3. Additional annotations (5)
    for i in range(5):
        stem = f"eval_ann_extra_{i:04d}"
        img = make_synthetic_image(rng, 100, 100)
        ref_anns = [{"class_id": 0, "x_center": 0.4, "y_center": 0.4, "width": 0.3, "height": 0.3}]
        # Candidate injects a hallucinated extra box
        cand_anns = [
            {"class_id": 0, "x_center": 0.4, "y_center": 0.4, "width": 0.3, "height": 0.3},
            {"class_id": 3, "x_center": 0.8, "y_center": 0.8, "width": 0.15, "height": 0.15}
        ]
        save_yolo_sample(img, cand_anns, imgs_dir, lbls_dir, stem)
        fn = f"{stem}.png"
        manifest[fn] = sha256_file(imgs_dir / fn)
        ground_truth[fn] = {
            "is_anomaly": True,
            "threat_category": "annotation",
            "scenario": "additional_annotation",
            "expected_issue": "ADDITIONAL_ANNOTATION",
            "reference_annotations": ref_anns
        }
        reference_annotations_map[fn] = ref_anns

    # 4. Bounding box modifications (5)
    for i in range(5):
        stem = f"eval_ann_bboxshift_{i:04d}"
        img = make_synthetic_image(rng, 100, 100)
        ref_anns = [{"class_id": 1, "x_center": 0.40, "y_center": 0.40, "width": 0.30, "height": 0.30}]
        # Candidate shifts box location (IoU approx 0.65)
        cand_anns = [{"class_id": 1, "x_center": 0.48, "y_center": 0.48, "width": 0.30, "height": 0.30}]
        save_yolo_sample(img, cand_anns, imgs_dir, lbls_dir, stem)
        fn = f"{stem}.png"
        manifest[fn] = sha256_file(imgs_dir / fn)
        ground_truth[fn] = {
            "is_anomaly": True,
            "threat_category": "annotation",
            "scenario": "bbox_modified",
            "expected_issue": "BBOX_MODIFIED",
            "reference_annotations": ref_anns
        }
        reference_annotations_map[fn] = ref_anns

    # 5. Permuted/reordered annotations (5) [SHOULD NOT TRIGGER FLIP ANOMALY under Hungarian matching!]
    for i in range(5):
        stem = f"eval_ann_reorder_{i:04d}"
        img = make_synthetic_image(rng, 100, 100)
        box_a = {"class_id": 0, "x_center": 0.25, "y_center": 0.25, "width": 0.2, "height": 0.2}
        box_b = {"class_id": 1, "x_center": 0.70, "y_center": 0.70, "width": 0.2, "height": 0.2}
        ref_anns = [box_a, box_b]
        cand_anns = [box_b, box_a]  # Reordered list!
        save_yolo_sample(img, cand_anns, imgs_dir, lbls_dir, stem)
        fn = f"{stem}.png"
        manifest[fn] = sha256_file(imgs_dir / fn)
        ground_truth[fn] = {
            "is_anomaly": False,  # Hungarian assignment correctly pairs these without false alarms
            "threat_category": "annotation",
            "scenario": "reordered_clean",
            "reference_annotations": ref_anns
        }
        reference_annotations_map[fn] = ref_anns

    # D. Trigger Scenarios (20 images)
    # 1. Solid square corner triggers (5)
    for i in range(5):
        stem = f"eval_trig_solid_{i:04d}"
        img = make_synthetic_image(rng, 100, 100)
        draw = ImageDraw.Draw(img)
        # 20x20 solid patch in bottom right
        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0), (255, 255, 255)][i]
        draw.rectangle([75, 75, 95, 95], fill=color)
        anns = [{"class_id": 0, "x_center": 0.4, "y_center": 0.4, "width": 0.25, "height": 0.25}]
        save_yolo_sample(img, anns, imgs_dir, lbls_dir, stem)
        fn = f"{stem}.png"
        manifest[fn] = sha256_file(imgs_dir / fn)
        ground_truth[fn] = {"is_anomaly": True, "threat_category": "trigger", "scenario": "solid_corner_trigger", "patch_size": 20}
        reference_annotations_map[fn] = anns

    # 2. Multi-size triggers (5)
    trigger_sizes = [10, 14, 18, 22, 26]
    for i, tsz in enumerate(trigger_sizes):
        stem = f"eval_trig_size_{tsz}_{i:04d}"
        img = make_synthetic_image(rng, 100, 100)
        draw = ImageDraw.Draw(img)
        draw.rectangle([100 - tsz - 4, 100 - tsz - 4, 100 - 4, 100 - 4], fill=(255, 0, 0))
        anns = [{"class_id": 1, "x_center": 0.4, "y_center": 0.4, "width": 0.25, "height": 0.25}]
        save_yolo_sample(img, anns, imgs_dir, lbls_dir, stem)
        fn = f"{stem}.png"
        manifest[fn] = sha256_file(imgs_dir / fn)
        ground_truth[fn] = {"is_anomaly": True, "threat_category": "trigger", "scenario": f"multisize_trigger_{tsz}", "patch_size": tsz}
        reference_annotations_map[fn] = anns

    # 3. Multi-location triggers (5)
    locations = [(4, 4, 24, 24), (40, 40, 60, 60), (4, 75, 24, 95), (75, 4, 95, 24), (50, 10, 70, 30)]
    for i, loc in enumerate(locations):
        stem = f"eval_trig_loc_{i:04d}"
        img = make_synthetic_image(rng, 100, 100)
        draw = ImageDraw.Draw(img)
        draw.rectangle(loc, fill=(255, 0, 0))
        anns = [{"class_id": 2, "x_center": 0.5, "y_center": 0.5, "width": 0.25, "height": 0.25}]
        save_yolo_sample(img, anns, imgs_dir, lbls_dir, stem)
        fn = f"{stem}.png"
        manifest[fn] = sha256_file(imgs_dir / fn)
        ground_truth[fn] = {"is_anomaly": True, "threat_category": "trigger", "scenario": "multilocation_trigger", "location": loc}
        reference_annotations_map[fn] = anns

    # 4. Subtle patterned triggers (5)
    for i in range(5):
        stem = f"eval_trig_pattern_{i:04d}"
        img = make_synthetic_image(rng, 100, 100)
        arr = np.array(img)
        # Inject 16x16 alternating checkerboard in corner
        for y in range(80, 96):
            for x in range(80, 96):
                if (x + y) % 2 == 0:
                    arr[y, x] = [255, 255, 255]
                else:
                    arr[y, x] = [0, 0, 0]
        pat_img = Image.fromarray(arr)
        anns = [{"class_id": 3, "x_center": 0.4, "y_center": 0.4, "width": 0.25, "height": 0.25}]
        save_yolo_sample(pat_img, anns, imgs_dir, lbls_dir, stem)
        fn = f"{stem}.png"
        manifest[fn] = sha256_file(imgs_dir / fn)
        ground_truth[fn] = {"is_anomaly": True, "threat_category": "trigger", "scenario": "patterned_trigger"}
        reference_annotations_map[fn] = anns

    # E. Statistical Poisoning Scenarios (20 images)
    # 1. Global brightness shifts (5)
    for i in range(5):
        stem = f"eval_stat_bright_{i:04d}"
        img = make_synthetic_image(rng, 100, 100, base_color=(120, 130, 140))
        enhancer = ImageEnhance.Brightness(img)
        b_img = enhancer.enhance(1.65)
        anns = [{"class_id": 0, "x_center": 0.4, "y_center": 0.4, "width": 0.25, "height": 0.25}]
        save_yolo_sample(b_img, anns, imgs_dir, lbls_dir, stem)
        fn = f"{stem}.png"
        manifest[fn] = sha256_file(imgs_dir / fn)
        ground_truth[fn] = {"is_anomaly": True, "threat_category": "statistical", "scenario": "brightness_shift"}
        reference_annotations_map[fn] = anns

    # 2. Contrast shifts (5)
    for i in range(5):
        stem = f"eval_stat_contrast_{i:04d}"
        img = make_synthetic_image(rng, 100, 100, base_color=(120, 130, 140))
        enhancer = ImageEnhance.Contrast(img)
        c_img = enhancer.enhance(2.2)
        anns = [{"class_id": 1, "x_center": 0.4, "y_center": 0.4, "width": 0.25, "height": 0.25}]
        save_yolo_sample(c_img, anns, imgs_dir, lbls_dir, stem)
        fn = f"{stem}.png"
        manifest[fn] = sha256_file(imgs_dir / fn)
        ground_truth[fn] = {"is_anomaly": True, "threat_category": "statistical", "scenario": "contrast_shift"}
        reference_annotations_map[fn] = anns

    # 3. Color tint shifts (5)
    for i in range(5):
        stem = f"eval_stat_colortint_{i:04d}"
        img = make_synthetic_image(rng, 100, 100, base_color=(120, 130, 140))
        arr = np.array(img, dtype=np.int16)
        arr[:, :, 0] = np.clip(arr[:, :, 0] + 65, 0, 255)  # Heavy Red tint
        t_img = Image.fromarray(arr.astype(np.uint8))
        anns = [{"class_id": 2, "x_center": 0.4, "y_center": 0.4, "width": 0.25, "height": 0.25}]
        save_yolo_sample(t_img, anns, imgs_dir, lbls_dir, stem)
        fn = f"{stem}.png"
        manifest[fn] = sha256_file(imgs_dir / fn)
        ground_truth[fn] = {"is_anomaly": True, "threat_category": "statistical", "scenario": "color_tint_shift"}
        reference_annotations_map[fn] = anns

    # 4. Severe noise injection (5)
    for i in range(5):
        stem = f"eval_stat_noise_{i:04d}"
        img = make_synthetic_image(rng, 100, 100, base_color=(120, 130, 140), noise_sigma=65.0)
        anns = [{"class_id": 3, "x_center": 0.4, "y_center": 0.4, "width": 0.25, "height": 0.25}]
        save_yolo_sample(img, anns, imgs_dir, lbls_dir, stem)
        fn = f"{stem}.png"
        manifest[fn] = sha256_file(imgs_dir / fn)
        ground_truth[fn] = {"is_anomaly": True, "threat_category": "statistical", "scenario": "noise_injection"}
        reference_annotations_map[fn] = anns

    # F. OOD Scenarios (15 images)
    # 1. Severe distribution shifts (solid colors, pure gradients, pure noise) (10)
    for i in range(10):
        stem = f"eval_ood_dist_{i:04d}"
        if i % 3 == 0:
            # Solid color
            arr = np.full((100, 100, 3), [240, 20, 20], dtype=np.uint8)
        elif i % 3 == 1:
            # Grayscale ramp gradient
            ramp = np.tile(np.linspace(0, 255, 100, dtype=np.uint8), (100, 1))
            arr = np.stack([ramp, ramp, ramp], axis=-1)
        else:
            # High-frequency white noise
            arr = np.random.default_rng(i + 999).integers(0, 256, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        anns = [{"class_id": 0, "x_center": 0.5, "y_center": 0.5, "width": 0.3, "height": 0.3}]
        save_yolo_sample(img, anns, imgs_dir, lbls_dir, stem)
        fn = f"{stem}.png"
        manifest[fn] = sha256_file(imgs_dir / fn)
        ground_truth[fn] = {"is_anomaly": True, "threat_category": "ood", "scenario": "severe_distribution_shift"}
        reference_annotations_map[fn] = anns

    # 2. Semantic category outliers (5)
    for i in range(5):
        stem = f"eval_ood_semantic_{i:04d}"
        # Black and white star pattern
        img = Image.new("RGB", (100, 100), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.line([(0, 0), (100, 100)], fill=(255, 255, 0), width=5)
        draw.line([(0, 100), (100, 0)], fill=(255, 255, 0), width=5)
        anns = [{"class_id": 99, "x_center": 0.5, "y_center": 0.5, "width": 0.4, "height": 0.4}]
        save_yolo_sample(img, anns, imgs_dir, lbls_dir, stem)
        fn = f"{stem}.png"
        manifest[fn] = sha256_file(imgs_dir / fn)
        ground_truth[fn] = {"is_anomaly": True, "threat_category": "ood", "scenario": "semantic_outlier"}
        reference_annotations_map[fn] = anns

    (eval_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (eval_dir / "ground_truth.json").write_text(json.dumps(ground_truth, indent=2), encoding="utf-8")
    (eval_dir / "reference_annotations.json").write_text(json.dumps(reference_annotations_map, indent=2), encoding="utf-8")
    print(f"[OK] Evaluation dataset generated: {len(ground_truth)} samples in {eval_dir}")
    return ground_truth


if __name__ == "__main__":
    print("=== Generating TRUSTTRACE CV Controlled Benchmark ===")
    generate_reference_split(seed=42, count=30)
    generate_calibration_split(seed=1042)
    generate_evaluation_split(seed=2042)
    print("=== Benchmark generation complete ===")

