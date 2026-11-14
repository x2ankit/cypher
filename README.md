<div align="center">

# 🛡️ TrustTrace CV

### Deep Learning Integrity, Cryptographic Provenance & Adversarial Assurance Toolkit for Computer Vision Pipelines

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-005CED?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![Tests Passing](https://img.shields.io/badge/Tests-155%20Passed-success?style=for-the-badge&logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

<br/>

**[Overview](#-overview)** •
**[Key Features](#-key-features)** •
**[Ecosystem Comparison](#-how-it-compares)** •
**[System Architecture](#-system-architecture)** •
**[Empirical Benchmarks](#-empirical-benchmarks--scientific-validation)** •
**[Quick Start](#-quick-start-guide)** •
**[Python API](#-python-api-examples)** •
**[CLI Reference](#-command-line-interface-cli)** •
**[REST API](#-rest-api-reference)** •
**[Documentation](#-documentation-index)**

<br/>

<img src="assets/ui.png" alt="TrustTrace CV Dashboard" width="850" style="border-radius: 8px; box-shadow: 0 8px 24px rgba(0,0,0,0.15); border: 1px solid #e1e4e8;" />

*The TrustTrace CV interactive dashboard: Real-time mathematical audit traces, cryptographic verification, and offline forensic report generation.*

</div>

<br/>

---

## 📖 Overview

**TrustTrace CV** is an offline-first, model-agnostic assurance and cryptographic provenance toolkit designed specifically for safety-critical Computer Vision (CV) pipelines. Modern computer vision systems deployed in autonomous vehicles, robotics, healthcare diagnostics, and automated inspection are increasingly vulnerable to **data poisoning**, **Trojan backdoor insertion**, **model substitution**, and **silent inference log manipulation**.

Standard MLOps tools focus either on data validation or post-deployment monitoring without verifying cryptographic trust. **TrustTrace CV** bridges this security gap by delivering **zero-trust cryptographic verification**, **mathematically principled statistical heuristics**, and an **immutable hash-chained provenance ledger**—operating entirely **air-gapped** with zero external cloud dependencies.

---

## ✨ Key Features

### 📁 1. Dataset Integrity & Poisoning Defense
* **Exact Duplicate Flooding (`SEC-DS-001`)**: High-throughput cryptographic SHA-256 byte hashing to flag adversarial sample flooding and train/test leakage.
* **Perceptual Near-Duplicate Detection (`SEC-DS-001b`)**: 64-bit Difference Hash (dHash) with calibrated Hamming distance filtering to identify recompressed (JPEG quality shift), resized, or subtly filtered duplicates.
* **Label Flipping & BBox Integrity (`SEC-DS-002`)**: Optimal bipartite matching via the **Hungarian algorithm (Kuhn-Munkres)** with class affinity tie-breaking ($IoU \ge 0.50$). Dissected into `CLASS_CHANGE`, `BBOX_MODIFIED`, `MISSING_ANNOTATION`, and `ADDITIONAL_ANNOTATION`. Guaranteed 0 false positives under annotation reordering.
* **Backdoor Trigger Pattern Detection (`SEC-DS-003`)**: Sliding-window Shannon patch entropy analysis ($16 \times 16$ patches) detecting low-entropy synthetic overlays, BadNets triggers, and solid pixel watermarks.
* **Data Poisoning & Distribution Shift (`SEC-DS-004`)**: Robust, reference-calibrated Modified Z-scores using **Median and Median Absolute Deviation (MAD)** across color spaces, resisting heavy outlier masking.
* **Out-of-Distribution (OOD) Detection (`SEC-DS-005`)**: Reference-calibrated channel-covariance **Mahalanobis Distance** identifying semantic domain anomalies and corruptions.
* **Structural Schema Validation**: Offline parsers for **YOLO (v5/v8)** text coordinates and **COCO 1.0 JSON** schemas.

### 🧠 2. Model Integrity & Checkpoint Security
* **Cryptographic Manifest Hashing**: SHA-256 digest calculation for model weights, architectures, and associated assets.
* **Dual-Scheme Signature Verification**:
  * **Symmetric HMAC-SHA256**: High-speed, local tamper detection with timing-attack safe comparisons (`hmac.compare_digest`).
  * **Asymmetric RSA-2048 PKCS#1v15**: Enterprise-grade public-key attestation using X.509 PEM certificates.
* **Safe Deserialization (`weights_only=True`)**: Strict PyTorch unpickling protection preventing arbitrary code execution and Zip Slip exploits from untrusted checkpoint files.
* **Multi-Format Support**: Native structural introspection for **PyTorch (`.pt`, `.pth`)**, **TorchScript bytecode (`.ts`)**, and **ONNX graphs (`.onnx`)**.
* **Model Substitution Protection**: Deterministic parameter count and tensor structure verification preventing unauthorized model swaps.

### ⛓️ 3. Tamper-Evident Inference Provenance Ledger
* **Append-Only Cryptographic Chain**: Every logged inference record is cryptographically linked to the previous record (`previous_record_hash` $\to$ current HMAC), forming an immutable hash chain.
* **Strict Monotonic Sequencing**: Enforces gapless, strictly increasing `sequence_number` counters to instantly detect dropped, injected, or reordered frames.
* **Anti-Replay Nonce Verification**: Cryptographic UUID nonces prevent replay attacks from recorded telemetry streams.
* **Canonical JSON Serialization**: Deterministic key sorting (`sort_keys=True`) and compact whitespace elimination (`separators=(',', ':')`) preventing byte-level hashing ambiguities.

### 📊 4. Full-Stack Assurance & Forensic Reporting
* **Interactive Forensic Reports**: Generates self-contained, offline HTML audit reports with base64-embedded anomaly visualizations and complete audit logs.
* **Live REST API & Dashboard**: Asynchronous FastAPI service paired with a lightweight web UI providing real-time mathematical audit traces.
* **Zero-Trust Adapter Architecture**: Modular adapters for **Cleanlab**, **CleanVision**, **IBM ART**, **BackdoorBench**, **TrojAI**, **in-toto**, and **Cosign** that degrade honestly to `NOT_ASSESSED` in offline air-gapped environments without generating simulated passes.

---

## ⚖️ How It Compares

| Capability | TrustTrace CV | Deepchecks (Vision) | CleanVision | IBM ART | Sigstore / Cosign |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Primary Scope** | **End-to-End Cryptographic & Statistical Assurance** | ML Validation & Testing | Visual Data Quality | Adversarial Attacks & Robustness | Container / Artifact Signing |
| **Cryptographic Provenance** | ✅ **Native (HMAC + RSA hash chaining)** | ❌ None | ❌ None | ❌ None | ✅ Signature only |
| **Hungarian BBox & Label Audit** | ✅ **Optimal Bipartite Matching** | ⚠️ Generic Heuristics | ❌ None | ⚠️ Indirect | ❌ None |
| **Patch-Level Entropy Scans** | ✅ **Native ($16 \times 16$ Shannon Entropy)** | ❌ None | ❌ None | ⚠️ Complex Defense | ❌ None |
| **Perceptual Near-Duplicates** | ✅ **64-bit dHash (Hamming Dist $\le 8$)** | ⚠️ Basic Duplicates | ✅ Visual Near-Dup | ❌ None | ❌ None |
| **Monotonic Inference Ledger** | ✅ **Gapless Sequence + Anti-Replay** | ❌ None | ❌ None | ❌ None | ❌ None |
| **Air-Gapped / Offline Native** | ✅ **100% Offline (No Cloud Calls)** | ⚠️ Requires Dependencies | ✅ Offline | ✅ Offline | ⚠️ Needs Rekor/Fulcio |
| **Safe Deserialization** | ✅ **Enforced (`weights_only=True`)** | ⚠️ Standard Pickling | ❌ N/A | ⚠️ Standard Loading | ❌ N/A |
| **Self-Contained Offline Reports** | ✅ **Interactive HTML (Base64 embed)** | ⚠️ HTML / Cloud | ❌ None | ❌ None | ❌ None |

---

## 🏗️ System Architecture

TrustTrace CV enforces an end-to-end pipeline verifying assets at every phase of the vision lifecycle:

```mermaid
flowchart TD
    subgraph INGESTION["1. Ingestion Layer"]
        A1["Raw CV Dataset (YOLO / COCO)"]
        A2["Model Artifact (.pt / .ts / .onnx)"]
        A3["Inference Stream (.jsonl)"]
    end

    subgraph ENGINE["2. TrustTrace Core Assurance Engine"]
        B1["Dataset Inspector & Schema Parser"]
        B2["Security Heuristics Suite"]
        B3["Safe Model Loader (weights_only=True)"]
        B4["Cryptographic Verifier (HMAC / RSA)"]
        B5["Inference Provenance Validator"]

        B1 --> B2
        B3 --> B4
    end

    subgraph CHECKS["3. Mathematical Verification Battery"]
        C1["Exact SHA-256 + Perceptual dHash"]
        C2["Hungarian IoU Bipartite Matcher"]
        C3["Sliding Shannon Patch Entropy"]
        C4["Reference-Calibrated Robust MAD"]
        C5["Channel Mahalanobis Distance"]
        C6["Hash Chaining & Sequence Monotonicity"]
        
        B2 --> C1
        B2 --> C2
        B2 --> C3
        B2 --> C4
        B2 --> C5
        B5 --> C6
    end

    subgraph AUDIT["4. Tamper-Evident Storage & Reporting"]
        D1[("SQLite Audit Event Store (Append-Only)")]
        D2["Evidence Correlator & Verdict Engine"]
        D3["Self-Contained Offline HTML Report"]
        D4["Structured JSON Audit Log"]

        C1 & C2 & C3 & C4 & C5 & C6 & B4 --> D1
        D1 --> D2
        D2 --> D3
        D2 --> D4
    end
```

---

## 🔬 Empirical Benchmarks & Scientific Validation

TrustTrace CV enforces zero experimental leakage: thresholds are selected strictly on a dedicated **Calibration Split** and evaluated on an independent **Evaluation Benchmark Split** (130 evaluation samples, 30 reference baselines).

The table below reflects empirical results measured directly from our reproducible test harness (`scripts/run_benchmark.py`):

| Detector / Assurance Module | Threat Model Evaluated | Latency (ms) | Peak Memory | Precision | Recall | F1 Score | AUROC | AUPR | FPR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exact Duplicate (SHA-256)** | Adversarial Dataset Flooding | **1.13 ms** | 13.5 KB | 0.500 | 1.000 | 0.667 | — | — | 0.111 |
| **Perceptual dHash (Hamming $\le 8$)** | Recompression / Resizing Evasion | 367.43 ms | 127.5 KB | 0.563 | 0.900 | 0.692 | 0.742 | 0.578 | 0.467 |
| **Hungarian IoU Matching** | Label Flipping & BBox Alterations | **59.84 ms** | 26.7 KB | **1.000** | **1.000** | **1.000** | — | — | **0.000** |
| **Sliding Shannon Patch Entropy** | BadNets Solid Trigger Overlays | 938.31 ms | 318.7 KB | **1.000** | 0.400 | 0.571 | 0.870 | 0.864 | **0.000** |
| **Ref-Calibrated Robust MAD** | Feature Poisoning & Color Shift | 26.90 ms | 74.6 KB | 0.714 | **1.000** | 0.833 | **0.987** | **0.977** | 0.267 |
| **Ref-Calibrated Mahalanobis** | Out-of-Distribution (OOD) Samples | 19.66 ms | 23.2 KB | 0.692 | 0.600 | 0.643 | 0.796 | 0.783 | 0.133 |

> [!NOTE]
> * **Zero False Positives on Label Permutations**: The Hungarian bipartite matcher achieves an F1-score of 1.000 with 0 false positives when clean annotations are merely reordered.
> * **Robust Outlier Resistance**: Reference-calibrated MAD achieves **0.987 AUROC**, substantially outperforming uncalibrated global Z-scores that suffer from variance inflation when poisoned samples exceed 10% of the dataset.

---

## 🚦 Assurance Verdict Framework

Every subsystem check, API endpoint, and aggregated forensic report produces an explicit, unambiguous verdict:

```
                  ┌───────────────────────────────┐
                  │ Are findings empty or missing? │
                  └──────────────┬────────────────┘
                                 │
                     Yes ────────┴──────── No
                      │                    │
              [INCONCLUSIVE]      ┌─────────────────────────┐
                                  │ Cryptographic mismatch  │
                                  │ or tamper detected?     │
                                  └───────────┬─────────────┘
                                              │
                                  Yes ────────┴──────── No
                                   │                    │
                                [FAIL]         ┌─────────────────┐
                                               │ Any anomalies   │
                                               │ or shifts found?│
                                               └────────┬────────┘
                                                        │
                                            Yes ────────┴──────── No
                                             │                    │
                                   [ANOMALIES_DETECTED]        [PASS]
```

* **`PASS`**: All required cryptographic and statistical assurance checks executed with zero detected anomalies.
* **`ANOMALIES_DETECTED`**: Statistically significant deviations detected (e.g. perceptual duplicates, label discrepancies, trigger patterns, OOD outliers).
* **`FAIL`**: Definite security compromise (signature mismatch, broken hash chain, sequence gap, payload tampering).
* **`INCONCLUSIVE`**: Findings are empty or evidence is insufficient. Empty results never default to `PASS`.
* **`NOT_ASSESSED`**: A check could not execute because an optional dependency or reference baseline was not provided.

---

## 🚀 Quick Start Guide

### Prerequisites
* **Operating System**: Linux, macOS, or Windows
* **Python**: 3.10, 3.11, or 3.12
* **Hardware**: CPU standard; GPU optional for downstream model inference

### 1. Installation

Clone the repository and install dependencies in an isolated virtual environment:

```bash
# Clone repository
git clone https://github.com/x2ankit/trusttracecv.git
cd trusttracecv

# Create and activate conda or venv environment
conda create -n trusttrace python=3.12 -y
conda activate trusttrace

# Install core dependencies
pip install -r requirements.txt
```

### 2. Configure Cryptographic Environment

TrustTrace CV enforces zero hardcoded secrets. Copy the environment template and set secure keys:

```bash
cp .env.example .env
```

Edit `.env` to supply high-entropy secrets for cryptographic signing:
```ini
# Cryptographic Keys (Required for HMAC operations)
TRUSTTRACE_CV_HMAC_SECRET="replace_with_a_secure_random_hmac_key_for_inference"
TRUSTTRACE_CV_MANIFEST_SECRET="replace_with_a_secure_random_manifest_hmac_key"

# Optional: Path to RSA keys for asymmetric manifest signing and verification
TRUSTTRACE_PRIVATE_KEY_PATH="./private_key.pem"
TRUSTTRACE_PUBLIC_KEY_PATH="./public_key.pem"
```

To generate a standard RSA-2048 keypair:
```bash
openssl genrsa -out private_key.pem 2048
openssl rsa -in private_key.pem -pubout -out public_key.pem
```

### 3. Generate Synthetic Fixtures & Verify Setup

Generate reproducible offline test datasets and run the full test suite:

```bash
# Generate synthetic dataset and model fixtures
python scripts/generate_fixtures.py

# Run all 155 automated unit & integration tests
pytest tests/ -v
```

### 4. Launch the Server & Web Dashboard

Start the FastAPI backend with the interactive UI:

```bash
python run_server.py
```

Open your browser at **`http://localhost:8001`** (or `http://localhost:8000`) to inspect datasets, audit models, and review live cryptographic event logs.

---

## 💻 Python API Examples

### 1. Auditing a Computer Vision Dataset

```python
from pathlib import Path
from src.dataset.inspector import inspect_yolo_dataset
from src.dataset.security_checks import (
    check_duplicate_flooding,
    check_label_flipping,
    check_trigger_patterns,
    check_data_poisoning,
)

dataset_path = Path("data/fixtures/clean")
ref_annotations = Path("data/fixtures/clean/annotations_reference.json")

# 1. Parse YOLO dataset
dataset_summary = inspect_yolo_dataset(dataset_path)

# 2. Execute statistical heuristics
findings = []
findings.extend(check_duplicate_flooding(dataset_path))
findings.extend(check_trigger_patterns(dataset_path))
findings.extend(check_data_poisoning(dataset_path))
findings.extend(check_label_flipping(dataset_path, reference_annotations_path=ref_annotations))

print(f"Dataset audited: {dataset_summary.total_images} images. Findings: {len(findings)}")
```

### 2. Cryptographic Model Manifest & Verification

```python
from pathlib import Path
from src.models.integrity import inspect_model, create_manifest, verify_manifest_signature

model_path = Path("models/fixtures/dummy_detector.pt")
manifest_path = Path("models/fixtures/manifest.json")

# 1. Create a signed manifest
manifest = create_manifest(
    model_dir=model_path.parent,
    output_path=manifest_path,
    private_key_path=Path("private_key.pem")
)

# 2. Inspect model format safely (weights_only=True)
inspection = inspect_model(model_path, manifest_path=manifest_path)
print(f"Model: {model_path.name} | Status: {inspection.status} | SHA-256: {inspection.sha256[:16]}...")
```

### 3. Recording and Verifying Inference Provenance

```python
from src.inference.provenance import record_inference_event
from src.inference.verifier import verify_inference_chain

# 1. Log an inference event with HMAC cryptographic sealing
record = record_inference_event(
    image_id="frame_0001.png",
    model_id="yolov8n-coco",
    prediction={"classes": ["pedestrian"], "confidences": [0.94], "boxes": [[10, 20, 50, 80]]},
    execution_time_ms=14.2,
    secret_key="my_secret_signing_key"
)

# 2. Verify provenance integrity across the log
verification_findings = verify_inference_chain(
    log_path=Path("audit_events.sqlite"),
    secret_key="my_secret_signing_key"
)
```

---

## ⌨️ Command-Line Interface (CLI)

TrustTrace CV provides a rich, colorized CLI powered by Typer and Rich:

```bash
# Audit a YOLO or COCO dataset
python -m src.cli audit-dataset --dataset-dir data/fixtures/clean --format yolo

# Audit a model checkpoint against a reference manifest
python -m src.cli audit-model --model-path models/fixtures/dummy_detector.pt --manifest-path models/fixtures/manifest.json

# Verify an inference provenance log
python -m src.cli audit-inference --log-path data/fixtures/inference.jsonl

# Run the complete end-to-end pipeline and export a forensic report
python -m src.cli audit-all --dataset-dir data/fixtures/clean --model-path models/fixtures/dummy_detector.pt --output-dir reports/
```

---

## 🔌 REST API Reference

| Method | Endpoint | Description | Request Payload / Params |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/health` | Service health status check | None |
| `POST` | `/api/upload/dataset` | Upload ZIP dataset (YOLO/COCO) with Zip-Slip defense | `multipart/form-data` (`file`, `format`) |
| `POST` | `/api/audit/dataset` | Trigger asynchronous dataset security audit | `{"dataset_path": "...", "format": "yolo"}` |
| `POST` | `/api/audit/model` | Trigger model structural & cryptographic audit | `{"model_path": "...", "manifest_path": "..."}` |
| `POST` | `/api/inference/log` | Log an inference prediction with HMAC provenance | `{"image_id": "...", "model_id": "...", "prediction": {...}}` |
| `GET` | `/api/audit/status` | Polling endpoint for active background audits | None |
| `GET` | `/api/audit/events/{id}` | Chronological audit trail from SQLite ledger | Path parameter `id` (sanitized alphanumeric) |
| `GET` | `/api/audit/records/{id}` | Detailed image-level anomaly records | Path parameter `id` |
| `GET` | `/api/reports/html/{id}` | Download self-contained offline HTML forensic report | Path parameter `id` |
| `GET` | `/api/reports` | List all historical JSON audit reports | None |

---

## 📂 Project Structure

```
trusttracecv/
├── assets/                     # Architectural diagrams, dashboard previews
│   ├── ui.png
│   └── ui_screenshot.jpg
├── data/                       # Dataset directories (ignored by git)
│   ├── benchmark/              # Calibration & evaluation splits
│   └── fixtures/               # Synthetic YOLO/COCO testing fixtures
├── docs/                       # Technical specifications & guides
│   ├── API_REFERENCE.md        # Comprehensive REST API specifications
│   ├── ARCHITECTURE.md         # Subsystem dataflows & architectural design
│   ├── DATASET_TESTING.md      # Mathematical formulations & benchmark protocols
│   ├── DEVELOPMENT.md          # Developer setup & contribution guidelines
│   └── SECURITY_AND_THREAT_MODEL.md # Threat models, cryptanalysis & boundaries
├── models/                     # Model weights & manifests (ignored by git)
│   └── fixtures/               # Synthetic PyTorch/TorchScript test models
├── reports/                    # Generated forensic JSON & HTML audit outputs
│   └── benchmark_results.json  # Empirical benchmark results
├── scripts/                    # Generation and benchmark utilities
│   ├── generate_fixtures.py    # Synthetic fixture generator
│   ├── generate_benchmark.py   # Benchmark split generator
│   ├── run_benchmark.py        # Scientific benchmark runner
│   └── demo_behavior.py        # End-to-end behavioral fingerprinting demo
├── src/                        # Core TrustTrace CV library
│   ├── api/                    # FastAPI backend & SQLite audit DB
│   │   ├── app.py
│   │   └── audit_db.py
│   ├── dataset/                # Dataset inspection & statistical security checks
│   │   ├── inspector.py
│   │   └── security_checks.py
│   ├── evidence/               # Evidence correlation & verdict calculation
│   │   └── correlator.py
│   ├── inference/              # Inference HMAC provenance & replay verifier
│   │   ├── provenance.py
│   │   └── verifier.py
│   ├── integrations/           # Zero-trust adapters (Cleanlab, ART, in-toto, etc.)
│   ├── models/                 # Model integrity, manifests & runtime probes
│   │   ├── behavioral_fingerprint.py
│   │   ├── controlled_transformations.py
│   │   ├── integrity.py
│   │   ├── runtime.py
│   │   └── trigger_probe.py
│   ├── reporting/              # HTML & JSON report generators
│   │   ├── html_generator.py
│   │   └── report_generator.py
│   ├── shift/                  # Distribution shift detection
│   │   └── distribution.py
│   └── ui/                     # Lightweight frontend dashboard
│       ├── index.html
│       ├── app.js
│       └── style.css
├── tests/                      # Automated test suite (155 pytest cases)
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_audit_security_fixes.py
│   ├── test_behavior.py
│   ├── test_dataset.py
│   ├── test_dataset_benchmarking.py
│   ├── test_inference.py
│   ├── test_models.py
│   └── test_report.py
├── .env.example                # Template for cryptographic secrets
├── .gitignore                  # Git ignore rules
├── LICENSE                     # MIT License
├── pytest.ini                  # Pytest configuration
├── requirements.txt            # Python dependencies
└── run_server.py               # Server entry point
```

---

## 🔒 Threat Model & Security Boundaries

TrustTrace CV addresses specific adversarial threats across the ML pipeline:

| Subsystem | Defended Threats | Out of Scope / Explicit Limitations |
| :--- | :--- | :--- |
| **Dataset Assurance** | Exact sample flooding, recompressed near-duplicates, label flipping, bounding-box shifts, synthetic solid triggers (BadNets), color distribution poisoning, semantic domain outliers. | Invisible adversarial perturbations with zero pixel-level statistical signature ($L_\infty \le 1/255$). |
| **Model Security** | Checkpoint tampering, model substitution, unauthorized fine-tuning, malicious pickle deserialization exploits. | Dynamic runtime memory tampering within secure hardware enclaves. |
| **Inference Provenance** | Prediction tampering, frame dropping, record reordering, telemetry replay attacks. | Physical camera sensor spoofing prior to digitized frame ingestion. |
| **Storage & API** | Path traversal (Zip Slip), database tampering, unsigned audit records, timing attacks. | Physical write-once hardware storage (WORM) enforcement. |

---

## 🧪 Testing & Quality Assurance

TrustTrace CV enforces test-driven quality assurance. The entire suite runs 100% offline using synthetic mathematical fixtures:

```bash
# Execute the full pytest suite
pytest tests/ -v

# Run with test coverage report
pytest --cov=src tests/
```

```
================================ test session starts =================================
platform win32 -- Python 3.12.14, pytest-8.3.3
rootdir: trusttracecv
collected 155 items

tests/test_api.py .........................                                    [ 16%]
tests/test_audit_security_fixes.py ........................................... [ 44%]
tests/test_behavior.py ........                                                [ 49%]
tests/test_dataset.py .......................                                  [ 64%]
tests/test_dataset_benchmarking.py ....................                        [ 77%]
tests/test_inference.py .................                                      [ 88%]
tests/test_models.py .................                                         [ 99%]
tests/test_report.py .                                                         [100%]

================================= 155 passed in 20.30s ================================
```

---

## 📚 Documentation Index

For complete architectural specifications, threat analyses, and development guides:
* 🏛️ **[Architecture Specification](docs/ARCHITECTURE.md)**: Subsystem dataflows, state machines, and integration adapters.
* 🛡️ **[Security & Threat Model](docs/SECURITY_AND_THREAT_MODEL.md)**: Detailed cryptanalysis, timing attack mitigations, and deserialization boundaries.
* 📐 **[Dataset Testing & Benchmarking](docs/DATASET_TESTING.md)**: Mathematical formulations (Hungarian matching, Shannon entropy, Mahalanobis distance) and evaluation protocols.
* 📡 **[REST API Reference](docs/API_REFERENCE.md)**: Endpoint documentation, parameters, and payloads.
* 💻 **[Development Guide](docs/DEVELOPMENT.md)**: Local development setup, code conventions, and testing workflows.

---

## 📄 License & Citation

TrustTrace CV is released under the **[MIT License](LICENSE)**.

```bibtex
@software{trusttracecv2026,
  author = {Ankit Tripathy and Contributors},
  title = {TrustTrace CV: Deep Learning Integrity, Cryptographic Provenance & Adversarial Assurance Toolkit for Computer Vision Pipelines},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/x2ankit/trusttracecv}
}
```

<div align="center">
  <sub>Engineered for Zero-Trust, Air-Gapped Computer Vision Assurance.</sub>
</div>
