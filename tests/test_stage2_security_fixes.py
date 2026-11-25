"""
tests/test_stage2_security_fixes.py

Comprehensive regression tests for Stage 2 security, reliability, and integrity fixes:
1. HTML report non-existent audit ID returns 404 and does not leak unrelated reports.
2. Benchmark runner does not execute runtime pip install subprocess.
3. Path traversal attacks on inference_log in /api/audit/full are blocked (Windows & POSIX).
4. Audit event hash chain verification detects valid chains, field tampering, and sequence breaks.
5. Background task state tracking accurately reflects lifecycle states and per-task status.
6. HMAC authentication is verified before inference provenance records are accepted into SQLite.
"""

import os
import sys
import json
import sqlite3
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from src.api.app import app, safe_resolve, ROOT, TASK_REGISTRY, set_task_state, TaskState
from src.reporting.html_generator import generate_offline_html_report
from src.api.audit_db import log_event, verify_audit_chain, get_db_path, init_db
from src.inference.verifier import record_inference
from src.inference.provenance import record_inference_event

client = TestClient(app)
TEST_SECRET = b"test-key-32-bytes-long-for-hmac!"


# ===========================================================================
# 1. HTML Report Fallback Fix & Regression Tests
# ===========================================================================

def test_html_report_nonexistent_returns_404():
    """Verify that requesting a nonexistent audit_id returns 404 rather than serving a fallback report."""
    fake_id = "non_existent_fake_audit_id_99999"
    resp = client.get(f"/api/reports/html/{fake_id}")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_html_generator_raises_file_not_found():
    """Directly test that generate_offline_html_report raises FileNotFoundError for missing reports."""
    with pytest.raises(FileNotFoundError, match="Report not found for audit_id"):
        generate_offline_html_report("non_existent_fake_audit_id_88888")


# ===========================================================================
# 2. Benchmark Runner Pip Removal Regression Test
# ===========================================================================

def test_benchmark_runner_raises_import_error_without_pip(monkeypatch):
    """Verify benchmark_runner raises clean ImportError if dependency is absent, without calling pip."""
    import src.reporting.benchmark_runner as br
    
    # Simulate missing pytest_jsonreport
    monkeypatch.setitem(sys.modules, "pytest_jsonreport", None)
    
    with pytest.raises(ImportError, match="pytest-json-report is required"):
        br.run_benchmarks()


# ===========================================================================
# 3. Path Traversal Defense Regression Tests (Windows & POSIX)
# ===========================================================================

@pytest.mark.parametrize("malicious_path", [
    "C:/Windows/System32/drivers/etc/hosts",
    "C:\\Windows\\win.ini",
    "../../../../etc/passwd",
    "..\\..\\..\\sensitive.log",
    "/etc/shadow",
])
def test_inference_path_traversal_blocked_in_full_audit(malicious_path):
    """Verify that Windows absolute paths and directory traversal payloads are blocked in /api/audit/full."""
    resp = client.post("/api/audit/full", json={
        "dataset_path": "",
        "model_path": "",
        "inference_log": malicious_path
    })
    # Must reject traversal with 403 Access Denied
    assert resp.status_code == 403
    assert "path traversal" in resp.json()["detail"].lower()


# ===========================================================================
# 4. Audit Event Hash Chain Verification & Tampering Tests
# ===========================================================================

def test_audit_chain_verification_lifecycle(tmp_path, monkeypatch):
    """Verify audit hash chain verification detects valid chains, tampered fields, and sequence breaks."""
    test_db = tmp_path / "chain_test.sqlite"
    monkeypatch.setenv("TRUSTTRACE_AUDIT_DB", str(test_db))
    init_db()

    audit_id = "test-chain-audit-001"

    # 1. Log 3 chained events
    log_event(audit_id, "Stage1", "Op1", {"k": "v1"}, "f1()", {}, "PASS", "t1", "DEC1", "EV1", 10.0, "RUNNING")
    log_event(audit_id, "Stage2", "Op2", {"k": "v2"}, "f2()", {}, "PASS", "t2", "DEC2", "EV2", 20.0, "RUNNING")
    log_event(audit_id, "Stage3", "Op3", {"k": "v3"}, "f3()", {}, "PASS", "t3", "DEC3", "EV3", 30.0, "PASS")

    # 2. Verify valid chain
    result = verify_audit_chain(audit_id)
    assert result["is_valid"] is True
    assert result["total_events"] == 3
    assert len(result["anomalies"]) == 0
    assert result["last_hash"] != ""

    # Verify via API endpoint
    api_resp = client.get(f"/api/audit/verify-chain/{audit_id}")
    assert api_resp.status_code == 200
    assert api_resp.json()["is_valid"] is True

    # 3. Tamper with intermediate record in SQLite
    conn = sqlite3.connect(test_db)
    cursor = conn.cursor()
    cursor.execute("UPDATE audit_events SET result = 'TAMPERED' WHERE audit_id = ? AND sequence_number = 2", (audit_id,))
    conn.commit()
    conn.close()

    # Chain verification MUST fail
    tampered_result = verify_audit_chain(audit_id)
    assert tampered_result["is_valid"] is False
    assert any("Hash mismatch" in a for a in tampered_result["anomalies"])

    # 4. Test sequence gap detection
    audit_id_gap = "test-chain-gap-002"
    log_event(audit_id_gap, "Stage1", "Op1", {}, "", {}, "PASS", "", "", "", 1.0, "PASS")
    log_event(audit_id_gap, "Stage2", "Op2", {}, "", {}, "PASS", "", "", "", 1.0, "PASS")
    log_event(audit_id_gap, "Stage3", "Op3", {}, "", {}, "PASS", "", "", "", 1.0, "PASS")

    conn = sqlite3.connect(test_db)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM audit_events WHERE audit_id = ? AND sequence_number = 2", (audit_id_gap,))
    conn.commit()
    conn.close()

    gap_result = verify_audit_chain(audit_id_gap)
    assert gap_result["is_valid"] is False
    assert any("Sequence gap" in a for a in gap_result["anomalies"])


# ===========================================================================
# 5. Background Task Lifecycle State Tracking Tests
# ===========================================================================

def test_task_state_tracking_lifecycle():
    """Verify task registry updates state and endpoint queries report accurate lifecycle information."""
    task_id = "test-task-lifecycle-99"

    # 1. Non-existent task returns 404
    resp_404 = client.get(f"/api/audit/status?audit_id=non_existent_task_123")
    assert resp_404.status_code == 404

    # 2. Register task as PENDING
    set_task_state(task_id, TaskState.PENDING, stage="Queued", detail="Task is waiting.")
    status_pending = client.get(f"/api/audit/status?audit_id={task_id}").json()
    assert status_pending["status"] == "PENDING"
    assert status_pending["stage"] == "Queued"

    # 3. Transition to RUNNING
    set_task_state(task_id, TaskState.RUNNING, stage="Processing", detail="Auditing images.")
    status_running = client.get(f"/api/audit/status?audit_id={task_id}").json()
    assert status_running["status"] == "RUNNING"
    assert status_running["stage"] == "Processing"

    # 4. Transition to COMPLETED
    set_task_state(task_id, TaskState.COMPLETED, stage="Done", detail="Audit completed.")
    status_completed = client.get(f"/api/audit/status?audit_id={task_id}").json()
    assert status_completed["status"] == "COMPLETED"
    assert status_completed["stage"] == "Done"

    # 5. Generic /api/audit/status returns Idle when no active tasks are running
    generic_status = client.get("/api/audit/status").json()
    assert generic_status["status"] == "Idle"
    assert generic_status["active_tasks"] == 0
    assert generic_status["latest_task"]["status"] == "COMPLETED"


def test_task_registry_retention_strategy():
    """Verify that conservative retention prunes terminal tasks without evicting active tasks."""
    # Register 2 active tasks
    active_1 = "active_task_1"
    active_2 = "active_task_2"
    set_task_state(active_1, TaskState.RUNNING, stage="Working")
    set_task_state(active_2, TaskState.PENDING, stage="Queued")

    # Add 220 completed tasks (exceeding MAX_TERMINAL_TASKS = 200)
    for i in range(220):
        t_id = f"finished_task_{i}"
        set_task_state(t_id, TaskState.COMPLETED, stage="Done")

    # Active tasks MUST still exist
    assert active_1 in TASK_REGISTRY
    assert active_2 in TASK_REGISTRY
    assert TASK_REGISTRY[active_1]["status"] == "RUNNING"
    assert TASK_REGISTRY[active_2]["status"] == "PENDING"

    # Terminal tasks count must be capped at 200
    terminal_count = sum(1 for t in TASK_REGISTRY.values() if t["status"] in ("COMPLETED", "FAILED"))
    assert terminal_count <= 200


# ===========================================================================
# 6. HMAC Signature Authentication in Inference Provenance
# ===========================================================================

def test_hmac_authentication_enforced(tmp_path, monkeypatch):
    """Verify that records with invalid, missing, or tampered HMAC signatures are rejected."""
    test_db = tmp_path / "prov_test.db"
    monkeypatch.setenv("TRUSTTRACE_PROVENANCE_DB", str(test_db))
    monkeypatch.setenv("TRUSTTRACE_CV_HMAC_SECRET", TEST_SECRET.decode("utf-8"))

    # 1. Valid signed record succeeds
    rec_valid = record_inference("img1", "model_yolo", [{"class_id": 0, "score": 0.9}], sequence_number=1, secret=TEST_SECRET)
    success, msg, r_hash = record_inference_event(rec_valid, secret=TEST_SECRET)
    assert success is True
    assert "successfully" in msg.lower()

    # 2. Tampered payload is rejected
    rec_tampered = record_inference("img2", "model_yolo", [{"class_id": 0, "score": 0.9}], sequence_number=2, secret=TEST_SECRET)
    rec_tampered["image_id"] = "evil_image" # tamper with image_id
    success_t, msg_t, _ = record_inference_event(rec_tampered, secret=TEST_SECRET)
    assert success_t is False
    assert msg_t == "INVALID_HMAC_SIGNATURE"

    # 3. Missing HMAC signature is rejected when secret is configured
    rec_unsigned = {
        "event_id": "test-unsigned-id",
        "image_id": "img3",
        "model_id": "model_yolo",
        "predictions": [],
        "sequence_number": 3,
    }
    success_u, msg_u, _ = record_inference_event(rec_unsigned, secret=TEST_SECRET)
    assert success_u is False
    assert msg_u == "MISSING_HMAC_SIGNATURE"


def test_hmac_former_test_keys_rejected_in_production(tmp_path, monkeypatch):
    """Verify that records signed with former test keys are rejected when a different production secret is configured."""
    test_db = tmp_path / "prov_prod_test.db"
    prod_secret = b"production_super_secret_isolated_key_32_bytes!"
    monkeypatch.setenv("TRUSTTRACE_PROVENANCE_DB", str(test_db))
    monkeypatch.setenv("TRUSTTRACE_CV_HMAC_SECRET", prod_secret.decode("utf-8"))

    # 1. Adversary attempts former test key 1 (b"test-secret-key")
    rec_fake_1 = record_inference("img1", "model1", [], secret=b"test-secret-key")
    success_1, msg_1, _ = record_inference_event(rec_fake_1)
    assert success_1 is False
    assert msg_1 == "INVALID_HMAC_SIGNATURE"

    # 2. Adversary attempts former test key 2 (b"test-key-32-bytes-long-for-hmac!")
    rec_fake_2 = record_inference("img2", "model1", [], secret=b"test-key-32-bytes-long-for-hmac!")
    success_2, msg_2, _ = record_inference_event(rec_fake_2)
    assert success_2 is False
    assert msg_2 == "INVALID_HMAC_SIGNATURE"

    # 3. Legitimate record signed with production secret succeeds
    rec_prod = record_inference("img3", "model1", [], secret=prod_secret)
    success_prod, msg_prod, _ = record_inference_event(rec_prod)
    assert success_prod is True
    assert "successfully" in msg_prod.lower()


# ===========================================================================
# 7. Audit Chain Nonexistent / Empty Handling Tests
# ===========================================================================

def test_empty_or_nonexistent_audit_chain_rejected(tmp_path, monkeypatch):
    """Verify that nonexistent or completely wiped audit records return is_valid=False and NO_EVENTS_FOUND."""
    test_db = tmp_path / "empty_chain_test.sqlite"
    monkeypatch.setenv("TRUSTTRACE_AUDIT_DB", str(test_db))
    init_db()

    # 1. Nonexistent audit ID
    res = verify_audit_chain("completely_nonexistent_audit_id")
    assert res["is_valid"] is False
    assert res["status"] == "NO_EVENTS_FOUND"
    assert res["total_events"] == 0
    assert len(res["anomalies"]) > 0

    # API endpoint also returns is_valid: False
    api_resp = client.get("/api/audit/verify-chain/completely_nonexistent_audit_id")
    assert api_resp.status_code == 200
    assert api_resp.json()["is_valid"] is False
    assert api_resp.json()["status"] == "NO_EVENTS_FOUND"

    # 2. Existing audit with records is valid
    audit_id = "test-chain-valid-nonempty"
    log_event(audit_id, "Stage1", "Op1", {}, "", {}, "PASS", "", "", "", 1.0, "PASS")
    res_valid = verify_audit_chain(audit_id)
    assert res_valid["is_valid"] is True
    assert res_valid["status"] == "VALID"
    assert res_valid["total_events"] == 1

    # 3. Delete all records for that audit in SQLite
    conn = sqlite3.connect(test_db)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM audit_events WHERE audit_id = ?", (audit_id,))
    conn.commit()
    conn.close()

    # Wiped audit must now return is_valid=False
    res_wiped = verify_audit_chain(audit_id)
    assert res_wiped["is_valid"] is False
    assert res_wiped["status"] == "NO_EVENTS_FOUND"


# ===========================================================================
# 8. README Example 2 End-to-End Verification Test
# ===========================================================================

def test_readme_example_2_works_end_to_end(tmp_path):
    """Verify that the exact code in README.md Example 2 executes without error against real fixtures."""
    from src.models.integrity import (
        build_manifest,
        save_manifest,
        load_manifest,
        verify_manifest,
        inspect_model,
    )

    model_path = Path("models/fixtures/dummy_detector.pt")
    manifest_path = tmp_path / "manifest.json"
    signing_secret = b"my_manifest_signing_key_32_bytes"

    # 1. Build and sign a cryptographic model manifest
    raw_manifest = build_manifest([model_path])
    save_manifest(raw_manifest, manifest_path, secret=signing_secret)

    # 2. Verify manifest signature
    loaded_manifest = load_manifest(manifest_path)
    is_valid, sig_type = verify_manifest(loaded_manifest, secret=signing_secret)
    assert is_valid is True
    assert sig_type == "HMAC"

    # 3. Inspect model format safely (weights_only=True) and verify against manifest
    inspection = inspect_model(model_path, reference_manifest=loaded_manifest)
    assert inspection["filename"] == "dummy_detector.pt"
    assert inspection["checks"]["hash_match"]["result"] == "PASS"
    assert len(inspection["sha256"]) == 64


# ===========================================================================
# 9. HMAC Fails Closed Without Configuration Tests
# ===========================================================================

def test_hmac_fails_closed_without_configuration(tmp_path, monkeypatch):
    """Verify that when no HMAC secret is configured:
    - Missing configuration causes verification to fail closed.
    - Explicitly supplied secret takes precedence over environment variable.
    """
    test_db = tmp_path / "prov_unconfigured.db"
    monkeypatch.setenv("TRUSTTRACE_PROVENANCE_DB", str(test_db))
    monkeypatch.delenv("TRUSTTRACE_CV_HMAC_SECRET", raising=False)

    # 1. Unsigned record without configuration fails closed
    rec_unsigned = {"event_id": "u1", "image_id": "img", "predictions": []}
    ok_1, msg_1, _ = record_inference_event(rec_unsigned)
    assert ok_1 is False
    assert msg_1 == "MISSING_HMAC_SIGNATURE"

    # 2. Signed record without configured secret fails closed with HMAC_SECRET_UNAVAILABLE
    rec_signed = {"event_id": "s1", "image_id": "img", "predictions": [], "hmac_sig": "dummy_sig"}
    ok_2, msg_2, _ = record_inference_event(rec_signed)
    assert ok_2 is False
    assert msg_2 == "HMAC_SECRET_UNAVAILABLE"

    # 3. Explicit secret takes precedence over env var
    env_secret = b"env_secret_key_32_bytes_long_1111!"
    explicit_secret = b"explicit_secret_key_32_bytes_long!"
    monkeypatch.setenv("TRUSTTRACE_CV_HMAC_SECRET", env_secret.decode("utf-8"))

    # Record created with explicit secret
    rec_explicit = record_inference("img_exp", "model1", [], secret=explicit_secret)
    # Passed with explicit secret: succeeds
    ok_3, msg_3, _ = record_inference_event(rec_explicit, secret=explicit_secret)
    assert ok_3 is True

    # Evaluated against env secret: fails because explicit secret was different
    rec_explicit_2 = record_inference("img_exp2", "model1", [], secret=explicit_secret)
    ok_4, msg_4, _ = record_inference_event(rec_explicit_2)  # uses env_secret
    assert ok_4 is False
    assert msg_4 == "INVALID_HMAC_SIGNATURE"


