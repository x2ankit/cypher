"""
src/reporting/report_generator.py

Assurance report generation for TRUSTTRACE CV.

Aggregates findings from dataset, model, and inference checks into a
structured assurance report. Reports include:
  - Overall assurance verdict (PASS / ANOMALIES_DETECTED / FAIL / INCONCLUSIVE)
  - Per-finding severity, confidence, evidence, and recommended action
  - Coverage statement (implemented, partial, unsupported checks)
  - Limitations section
  - Audit metadata (timestamp, random seed, module versions)

Output formats: Python dict (in-memory), JSON file.
"""

import json
import logging
import platform
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------

REPORT_SCHEMA_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Coverage statement
# ---------------------------------------------------------------------------

COVERAGE = {
    "implemented": [
        "SEC-DS-001: Duplicate flooding detection (SHA-256, exact match)",
        "SEC-DS-002: Label flipping detection (vs reference annotations)",
        "SEC-DS-003: Trigger pattern insertion (patch entropy heuristic)",
        "SEC-DS-004: Data poisoning via statistical outlier (Z-score on pixel stats)",
        "SEC-DS-005: OOD insertion via Mahalanobis distance on pixel distribution",
        "SEC-MDL-001: Model substitution / integrity via SHA-256 manifest",
        "SEC-MDL-002: Model structural validation (loadability, format, metadata)",
        "SEC-INF-001: Inference record tamper verification (HMAC-SHA256)",
        "SEC-INF-002: Inference replay detection (payload hash deduplication)",
        "SEC-INF-003: Backdoor-like behaviour indicator (confidence threshold)",
        "Dataset inspection: YOLO format (image, resolution, labels, bounding boxes)",
        "Dataset inspection: COCO format (annotations JSON, category distribution)",
    ],
    "partial": [
        "ONNX model inspection (requires onnx package; checker may fail on non-standard models)",
        "TorchScript loading (falls back gracefully; graph semantics not verified)",
        "Label flipping (requires trusted reference; if absent → NOT ASSESSED)",
        "Trigger detection (heuristic only; low confidence; sophisticated triggers may evade)",
    ],
    "unsupported": [
        "Universal backdoor detection (not mathematically feasible via static analysis alone)",
        "Semantic correctness of model weights",
        "Real-time streaming inference monitoring",
        "Near-duplicate image detection (perceptual hashing)",
        "Adversarial robustness evaluation (requires inference budget)",
        "Attribution / membership inference attacks",
    ],
    "assumptions": [
        "Dataset is provided in standard COCO or YOLO format.",
        "Model files are in PyTorch (.pt/.pth), TorchScript (.pt), or ONNX (.onnx) format.",
        "Inference records are produced by record_inference() or conform to that schema.",
        "A trusted reference annotation set exists for label-flipping detection.",
        "A reference manifest (SHA-256 per model file) exists for substitution detection.",
    ],
    "known_limitations": [
        "Statistical anomaly detection (Z-score, Mahalanobis, entropy) produces probabilistic "
        "signals; false positives and false negatives are expected.",
        "SHA-256 matching confirms file identity, NOT model safety.",
        "Trigger pattern search is entropy-based and cannot detect sophisticated textured triggers.",
        "HMAC signature verification requires the original signing key.",
        "Replay detection can be evaded by modifying any non-critical record field.",
        "Backdoor-like behaviour indicator (high confidence) cannot distinguish legitimate "
        "high confidence from triggered behaviour without ground truth.",
        "All checks assume honest and complete input data. Adversarially crafted inputs "
        "that comply with the expected format may not be detected.",
    ],
}


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------

def _normalize_result(finding: Dict[str, Any]) -> str:
    """Normalize finding result / status into standard assurance states."""
    res = finding.get("result") or finding.get("status") or "NOT_ASSESSED"
    res = str(res).strip().upper().replace(" ", "_")
    if res in ("NOT_ASSESSED", "N/A", "NONE"):
        return "NOT_ASSESSED"

    # Check for simulated evaluations
    obs = str(finding.get("observations", "")).lower()
    conf = str(finding.get("confidence_basis", "")).lower()
    lim = str(finding.get("limitations", "")).lower()
    if "simulated" in obs or "simulated" in conf or "simulated" in lim or res == "SIMULATED":
        return "SIMULATED"

    return res

def _overall_verdict(findings: List[Dict[str, Any]]) -> str:
    """
    Derive an overall assurance verdict from the list of findings.

    Explicit states:
      FAIL: A required security verification failed.
      ANOMALIES_DETECTED: Executed checks identified suspicious findings or distribution shifts.
      PASS: All required applicable checks executed and passed (requires genuine executed checks).
      NOT_ASSESSED: Checks could not be executed or dependencies/credentials were unavailable.
      INCONCLUSIVE: Evidence is empty or insufficient to establish an assurance result.
    """
    if not findings:
        return "INCONCLUSIVE"

    normalized = [_normalize_result(f) for f in findings]
    result_set = set(normalized)

    # 1. Deterministic or policy failure
    if "FAIL" in result_set:
        return "FAIL"

    # 2. Anomalies detected by executed checks
    if any(r in result_set for r in ("ANOMALY_DETECTED", "STATISTICAL_SHIFT", "FLAGGED", "WARNING")):
        return "ANOMALIES_DETECTED"

    # 3. Filter out simulated results (simulated never counts towards PASS)
    genuine_results = [r for r in normalized if r != "SIMULATED"]
    if not genuine_results:
        return "NOT_ASSESSED"

    genuine_set = set(genuine_results)

    # 4. If all genuine checks were NOT_ASSESSED
    if genuine_set == {"NOT_ASSESSED"}:
        return "NOT_ASSESSED"

    # 5. Inconclusive if insufficient evidence
    if "INCONCLUSIVE" in genuine_set and "PASS" not in genuine_set:
        return "INCONCLUSIVE"

    # 6. PASS only if genuine executed checks passed without anomaly or failure
    if "PASS" in genuine_set:
        return "PASS"

    return "NOT_ASSESSED"


def _severity_summary(findings: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {
        "CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "INFO": 0
    }
    for f in findings:
        sev = f.get("severity", "INFO")
        if sev in counts:
            counts[sev] += 1
    return counts


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def generate_report(
    dataset_findings: Optional[List[Dict[str, Any]]] = None,
    model_findings: Optional[List[Dict[str, Any]]] = None,
    inference_findings: Optional[List[Dict[str, Any]]] = None,
    dataset_summary: Optional[Dict[str, Any]] = None,
    model_summary: Optional[Dict[str, Any]] = None,
    audit_seed: int = 42,
    target_name: str = "Dataset Audit",
) -> Dict[str, Any]:
    """
    Assemble a complete assurance report.

    Args:
        dataset_findings:  List of finding dicts from dataset security checks.
        model_findings:    List of finding dicts from model integrity checks.
        inference_findings: List of finding dicts from inference verifier.
        dataset_summary:   High-level dataset statistics dict.
        model_summary:     High-level model inspection dict.
        audit_seed:        Random seed used for any probabilistic check.
        target_name:       Human-readable name for the audited subject.

    Returns:
        Complete structured report dict.
    """
    all_findings: List[Dict[str, Any]] = []
    if dataset_findings:
        all_findings.extend(dataset_findings)
    if model_findings:
        all_findings.extend(model_findings)
    if inference_findings:
        all_findings.extend(inference_findings)

    verdict = _overall_verdict(all_findings)
    severity_summary = _severity_summary(all_findings)

    import os
    import sys

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_id": str(uuid.uuid4()),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_name": target_name,
        "audit_seed": audit_seed,
        "system_info": {
            "platform": platform.platform(),
            "python_version": sys.version,
            "hostname": platform.node(),
        },
        "verdict": verdict,
        "severity_summary": severity_summary,
        "findings_count": len(all_findings),
        "sections": {
            "dataset": {
                "summary": dataset_summary or {},
                "findings": dataset_findings or [],
            },
            "models": {
                "summary": model_summary or {},
                "findings": model_findings or [],
            },
            "inference": {
                "findings": inference_findings or [],
            },
        },
        "coverage": COVERAGE,
        "evaluation_coverage": {
            "total_findings": len(all_findings),
            "executed_checks_count": len([f for f in all_findings if _normalize_result(f) in ("PASS", "ANOMALY_DETECTED", "FAIL", "STATISTICAL_SHIFT", "FLAGGED")]),
            "passed_checks_count": len([f for f in all_findings if _normalize_result(f) == "PASS"]),
            "anomaly_checks_count": len([f for f in all_findings if _normalize_result(f) in ("ANOMALY_DETECTED", "STATISTICAL_SHIFT", "FLAGGED")]),
            "failed_checks_count": len([f for f in all_findings if _normalize_result(f) == "FAIL"]),
            "not_assessed_checks_count": len([f for f in all_findings if _normalize_result(f) == "NOT_ASSESSED"]),
        },
        "confidence_basis": (
            "Deterministic checks (hash comparison, HMAC verification, exact duplicate "
            "detection) carry HIGH confidence. Statistical checks (Z-score, Mahalanobis "
            "distance, entropy) carry LOW to MEDIUM confidence and require human review. "
            "All anomaly findings represent candidates for investigation, not confirmed attacks."
        ),
        "limitations_summary": COVERAGE["known_limitations"],
        "recommended_actions": [
            f.get("recommended_action", "")
            for f in all_findings
            if f.get("result") in ("ANOMALY_DETECTED", "FAIL") or f.get("status") in ("FLAGGED", "FAIL")
            if f.get("recommended_action")
        ],
    }
    return report


def save_report(report: Dict[str, Any], output_path: Path) -> None:
    """Save a report dict to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    logger.info("Report saved to %s", output_path)


def load_report(report_path: Path) -> Dict[str, Any]:
    with open(report_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def generate_report_from_evidence(
    correlator_summary: Dict[str, Any],
    dataset_summary: Optional[Dict[str, Any]] = None,
    model_summary: Optional[Dict[str, Any]] = None,
    audit_seed: int = 42,
    target_name: str = "Full Assurance Audit"
) -> Dict[str, Any]:
    """
    Generate a report directly from an EvidenceCorrelator's summary.
    """
    import platform
    import sys
    
    findings = correlator_summary.get("findings", [])
    verdict = _overall_verdict(findings)
    severities = correlator_summary.get("severity_counts", {})

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_id": str(uuid.uuid4()),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_name": target_name,
        "audit_seed": audit_seed,
        "system_info": {
            "platform": platform.platform(),
            "python_version": sys.version,
            "hostname": platform.node(),
        },
        "verdict": verdict,
        "severity_summary": severities,
        "findings_count": correlator_summary.get("total_findings", 0),
        "sections": {
            "dataset": {
                "summary": dataset_summary or {},
            },
            "models": {
                "summary": model_summary or {},
            },
            "evidence": findings
        },
        "coverage": COVERAGE,
        "evaluation_coverage": {
            "total_findings": len(findings),
            "executed_checks_count": len([f for f in findings if _normalize_result(f) in ("PASS", "ANOMALY_DETECTED", "FAIL", "STATISTICAL_SHIFT", "FLAGGED")]),
            "passed_checks_count": len([f for f in findings if _normalize_result(f) == "PASS"]),
            "anomaly_checks_count": len([f for f in findings if _normalize_result(f) in ("ANOMALY_DETECTED", "STATISTICAL_SHIFT", "FLAGGED")]),
            "failed_checks_count": len([f for f in findings if _normalize_result(f) == "FAIL"]),
            "not_assessed_checks_count": len([f for f in findings if _normalize_result(f) == "NOT_ASSESSED"]),
        },
        "confidence_basis": (
            "Deterministic checks carry HIGH confidence. Statistical checks "
            "carry LOW to MEDIUM confidence and require human review. "
            "Evidence objects are correlated across DATASET, MODEL, INFERENCE, and SHIFT layers."
        ),
        "limitations_summary": COVERAGE["known_limitations"],
        "recommended_actions": [
            f.get("recommended_action", "")
            for f in findings
            if f.get("status") in ("FLAGGED", "FAIL") and f.get("recommended_action")
        ],
    }
    return report

