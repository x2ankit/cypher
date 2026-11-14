"""
tests/test_audit_security_fixes.py

Comprehensive tests verifying all Phase 1 audit fixes and security guarantees:
- Cryptographic integrity (manifest signing, RSA asymmetric verification, HMAC verification)
- Assurance verdict correctness and state transitions
- Inference provenance chain integrity (tamper detection, sequence validation, replay prevention)
- Honest integration behavior (no simulated PASS, honest NOT_ASSESSED)
- Dataset audit execution (label flipping with/without reference, OOD SEC-DS-005)
- Safe model loading (no unsafe pickle fallback)
- API security (Zip Slip prevention, path traversal rejection, audit ID validation)
"""

import json
import os
import shutil
import sqlite3
import tempfile
import uuid
import zipfile
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.dataset.security_checks import (
    check_duplicate_flooding,
    check_label_flipping,
    check_trigger_patterns,
    check_data_poisoning,
    check_ood_samples,
)
from src.inference.provenance import (
    init_db,
    record_inference_event,
    verify_provenance_chain,
)
from src.inference.verifier import (
    record_inference,
    verify_inference_record,
    check_inference_replay,
)
from src.integrations.art_adapter import run_art_robustness_probe
from src.integrations.intoto_adapter import InTotoEvidenceChain, generate_intoto_evidence
from src.models.integrity import (
    build_manifest,
    sign_manifest,
    verify_manifest,
    inspect_model,
    _try_load_pytorch,
)
from src.reporting.report_generator import (
    _overall_verdict,
    generate_report,
)


client = TestClient(app)
TEST_KEY = b"test-key-32-bytes-long-for-hmac!"


# ===========================================================================
# 1. Cryptographic Integrity Tests
# ===========================================================================

class TestCryptographicIntegrity:
    def test_manifest_signing_requires_secret(self, monkeypatch):
        monkeypatch.delenv("TRUSTTRACE_CV_MANIFEST_SECRET", raising=False)
        manifest = {"model.pt": "abc123hash"}
        with pytest.raises(ValueError, match="Signing secret required"):
            sign_manifest(manifest, secret=None)

    def test_manifest_verification_missing_secret(self, monkeypatch):
        monkeypatch.delenv("TRUSTTRACE_CV_MANIFEST_SECRET", raising=False)
        manifest = {"model.pt": "abc123hash", "hmac_sig": "a" * 64}
        is_valid, sig_type = verify_manifest(manifest, secret=None)
        assert is_valid is False
        assert sig_type == "HMAC_KEY_MISSING"

    def test_manifest_verification_valid_secret(self):
        manifest = {"model.pt": "abc123hash"}
        signed = sign_manifest(manifest, secret=TEST_KEY)
        assert "hmac_sig" in signed
        is_valid, sig_type = verify_manifest(signed, secret=TEST_KEY)
        assert is_valid is True
        assert sig_type == "HMAC"

    def test_manifest_verification_wrong_secret(self):
        manifest = {"model.pt": "abc123hash"}
        signed = sign_manifest(manifest, secret=TEST_KEY)
        is_valid, sig_type = verify_manifest(signed, secret=b"wrong-secret-key-32-bytes-long!")
        assert is_valid is False
        assert sig_type == "HMAC"

    def test_manifest_verification_tampered_payload(self):
        manifest = {"model.pt": "abc123hash"}
        signed = sign_manifest(manifest, secret=TEST_KEY)
        signed["model.pt"] = "tampered_hash_value"
        is_valid, sig_type = verify_manifest(signed, secret=TEST_KEY)
        assert is_valid is False
        assert sig_type == "HMAC"

    def test_manifest_rsa_asymmetric_verification(self, tmp_path):
        from cryptography.hazmat.primitives.asymmetric import rsa, padding
        from cryptography.hazmat.primitives import hashes, serialization
        import base64

        # Generate temporary test RSA key pair
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        public_key = private_key.public_key()
        pub_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        pub_path = tmp_path / "test_public_key.pem"
        pub_path.write_bytes(pub_pem)

        manifest = {"model_a.onnx": "hash123"}
        from src.models.integrity import canonicalize_manifest
        payload_bytes = canonicalize_manifest(manifest)
        raw_sig = private_key.sign(payload_bytes, padding.PKCS1v15(), hashes.SHA256())
        manifest["rsa_sig"] = base64.b64encode(raw_sig).decode("ascii")

        # Verify with public key
        is_valid, sig_type = verify_manifest(manifest, public_key_path=pub_path)
        assert is_valid is True
        assert sig_type == "ASYMMETRIC"

        # Verify fails if payload tampered
        manifest["model_a.onnx"] = "modified_hash"
        is_valid, sig_type = verify_manifest(manifest, public_key_path=pub_path)
        assert is_valid is False
        assert sig_type == "ASYMMETRIC"

        # Verify returns ASYMMETRIC_KEY_MISSING if public key absent
        is_valid, sig_type = verify_manifest(manifest, public_key_path=tmp_path / "ghost.pem")
        assert is_valid is False
        assert sig_type == "ASYMMETRIC_KEY_MISSING"

    def test_manifest_rsa_signing_and_verification_workflow(self, tmp_path):
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.primitives import serialization
        from src.models.integrity import sign_manifest_rsa

        # Generate RSA keypair and save to temporary files
        priv_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        priv_path = tmp_path / "private.pem"
        priv_path.write_bytes(
            priv_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
        )
        pub_path = tmp_path / "public.pem"
        pub_path.write_bytes(
            priv_key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
        )

        manifest = {"yolov8.pt": "sha256_mock_hash_value"}
        signed = sign_manifest_rsa(manifest, private_key_path=priv_path)
        assert "rsa_sig" in signed

        is_valid, sig_type = verify_manifest(signed, public_key_path=pub_path)
        assert is_valid is True
        assert sig_type == "ASYMMETRIC"

        # Missing key path raises FileNotFoundError
        with pytest.raises(FileNotFoundError):
            sign_manifest_rsa(manifest, private_key_path=tmp_path / "missing_key.pem")

    def test_inference_signing_requires_secret(self, monkeypatch):
        monkeypatch.delenv("TRUSTTRACE_CV_HMAC_SECRET", raising=False)
        with pytest.raises(ValueError, match="Signing key required"):
            record_inference("img1", "model1", [], secret=None)

    def test_inference_verification_missing_secret(self, monkeypatch):
        monkeypatch.delenv("TRUSTTRACE_CV_HMAC_SECRET", raising=False)
        rec = record_inference("img1", "model1", [], secret=TEST_KEY)
        finding = verify_inference_record(rec, secret=None)
        assert finding["result"] == "NOT_ASSESSED"
        assert "unavailable" in finding["description"]


# ===========================================================================
# 2. Assurance Verdict Correctness Tests
# ===========================================================================

class TestVerdictCorrectness:
    def test_empty_findings_is_inconclusive(self):
        assert _overall_verdict([]) == "INCONCLUSIVE"

    def test_all_not_assessed_is_not_assessed(self):
        findings = [
            {"result": "NOT_ASSESSED", "check_id": "SEC-DS-001"},
            {"result": "NOT ASSESSED", "check_id": "SEC-DS-002"},
        ]
        assert _overall_verdict(findings) == "NOT_ASSESSED"

    def test_simulated_evaluation_never_produces_pass(self):
        findings = [
            {
                "result": "PASS",
                "observations": "ART simulated robustness probe completed successfully.",
                "limitations": "Simulated evaluation for prototype.",
            }
        ]
        assert _overall_verdict(findings) != "PASS"
        assert _overall_verdict(findings) in ("NOT_ASSESSED", "INCONCLUSIVE")

    def test_genuine_pass_with_not_assessed(self):
        findings = [
            {"result": "PASS", "check_id": "SEC-DS-001"},
            {"result": "NOT_ASSESSED", "check_id": "SEC-DS-002"},
        ]
        assert _overall_verdict(findings) == "PASS"

    def test_anomaly_overrides_pass(self):
        findings = [
            {"result": "PASS", "check_id": "SEC-DS-001"},
            {"result": "ANOMALY_DETECTED", "check_id": "SEC-DS-003"},
        ]
        assert _overall_verdict(findings) == "ANOMALIES_DETECTED"

    def test_fail_overrides_anomaly_and_pass(self):
        findings = [
            {"result": "PASS", "check_id": "SEC-DS-001"},
            {"result": "ANOMALY_DETECTED", "check_id": "SEC-DS-003"},
            {"result": "FAIL", "check_id": "SEC-MDL-001"},
        ]
        assert _overall_verdict(findings) == "FAIL"

    def test_report_evaluation_coverage_metadata(self):
        findings = [
            {"result": "PASS", "check_id": "SEC-DS-001"},
            {"result": "NOT_ASSESSED", "check_id": "SEC-DS-002"},
            {"result": "ANOMALY_DETECTED", "check_id": "SEC-DS-003"},
        ]
        report = generate_report(dataset_findings=findings)
        assert report["verdict"] == "ANOMALIES_DETECTED"
        cov = report["evaluation_coverage"]
        assert cov["total_findings"] == 3
        assert cov["passed_checks_count"] == 1
        assert cov["anomaly_checks_count"] == 1
        assert cov["not_assessed_checks_count"] == 1

    def test_generate_report_from_evidence_has_coverage(self):
        from src.reporting.report_generator import generate_report_from_evidence
        summary = {
            "findings": [
                {"status": "PASS", "finding_type": "CHECK_A"},
                {"status": "FLAGGED", "finding_type": "CHECK_B"},
            ],
            "total_findings": 2,
            "severity_counts": {"INFO": 1, "MEDIUM": 1}
        }
        report = generate_report_from_evidence(summary)
        assert "evaluation_coverage" in report
        cov = report["evaluation_coverage"]
        assert cov["total_findings"] == 2
        assert cov["passed_checks_count"] == 1
        assert cov["anomaly_checks_count"] == 1


# ===========================================================================
# 3. Inference Record & Provenance Chain Integrity Tests
# ===========================================================================

class TestInferenceProvenanceChain:
    @pytest.fixture(autouse=True)
    def isolated_db(self, tmp_path):
        db_path = tmp_path / "prov_test.db"
        os.environ["TRUSTTRACE_PROVENANCE_DB"] = str(db_path)
        init_db()
        yield db_path
        if "TRUSTTRACE_PROVENANCE_DB" in os.environ:
            del os.environ["TRUSTTRACE_PROVENANCE_DB"]

    def test_valid_chain_passes_verification(self, isolated_db):
        last_hash = ""
        for i in range(3):
            rec = record_inference(f"img_{i}", "model_1", [], sequence_number=i, previous_event_hash=last_hash, secret=TEST_KEY)
            success, msg, r_hash = record_inference_event(rec)
            assert success is True
            last_hash = r_hash

        valid, status, anomalies = verify_provenance_chain()
        assert valid is True
        assert status == "CHAIN_VALID"
        assert len(anomalies) == 0

    def test_tampered_stored_field_detected(self, isolated_db):
        last_hash = ""
        for i in range(2):
            rec = record_inference(f"img_{i}", "model_1", [], sequence_number=i, previous_event_hash=last_hash, secret=TEST_KEY)
            _, _, r_hash = record_inference_event(rec)
            last_hash = r_hash

        # Tamper directly in the SQLite database row
        conn = sqlite3.connect(isolated_db)
        cursor = conn.cursor()
        cursor.execute("UPDATE provenance_events SET model_id = 'compromised_model' WHERE sequence_number = 1")
        conn.commit()
        conn.close()

        valid, status, anomalies = verify_provenance_chain()
        assert valid is False
        assert status == "CHAIN_BROKEN"
        assert any(a["issue"] == "FIELD_MODIFICATION" for a in anomalies)

    def test_broken_hash_link_detected(self, isolated_db):
        last_hash = ""
        for i in range(3):
            rec = record_inference(f"img_{i}", "model_1", [], sequence_number=i, previous_event_hash=last_hash, secret=TEST_KEY)
            _, _, r_hash = record_inference_event(rec)
            last_hash = r_hash

        # Tamper with previous_event_hash in middle event
        conn = sqlite3.connect(isolated_db)
        cursor = conn.cursor()
        cursor.execute("UPDATE provenance_events SET previous_event_hash = 'corrupted_hash' WHERE sequence_number = 1")
        conn.commit()
        conn.close()

        valid, status, anomalies = verify_provenance_chain()
        assert valid is False
        assert status == "CHAIN_BROKEN"
        assert any(a["issue"] == "BROKEN_LINK" for a in anomalies)

    def test_sequence_out_of_order_detected(self, isolated_db):
        # Insert event with invalid sequence inversion
        rec1 = record_inference("img_0", "model_1", [], sequence_number=5, secret=TEST_KEY)
        record_inference_event(rec1)
        rec2 = record_inference("img_1", "model_1", [], sequence_number=2, previous_event_hash=rec1["payload_hash"], secret=TEST_KEY)
        record_inference_event(rec2)

        valid, status, anomalies = verify_provenance_chain()
        assert valid is False
        assert any(a["issue"] == "SEQUENCE_OUT_OF_ORDER" for a in anomalies)

    def test_record_inference_event_rejects_tampered_payload_hash(self):
        rec = record_inference("img_0", "model_1", [{"class_id": 0, "score": 0.8}], secret=TEST_KEY)
        rec["payload_hash"] = "fake_hash_not_matching_content"
        success, msg, _ = record_inference_event(rec)
        assert success is False
        assert msg == "PAYLOAD_HASH_MISMATCH"

    def test_nonce_reuse_rejected(self, isolated_db):
        rec1 = record_inference("img_0", "model_1", [], secret=TEST_KEY)
        success1, _, _ = record_inference_event(rec1)
        assert success1 is True

        # Second record with same nonce but different image/timestamp
        rec2 = record_inference("img_different", "model_1", [], secret=TEST_KEY)
        rec2["nonce"] = rec1["nonce"]
        from src.inference.verifier import canonicalize_inference_payload
        import hashlib, hmac
        # Re-derive payload hash with reused nonce
        cb = canonicalize_inference_payload(rec2)
        rec2["payload_hash"] = hashlib.sha256(cb).hexdigest()
        rec2["hmac_sig"] = hmac.new(TEST_KEY, cb, hashlib.sha256).hexdigest()

        success2, msg2, _ = record_inference_event(rec2)
        assert success2 is False
        assert msg2 == "REPLAY_DETECTED"

    def test_tampered_non_model_field_detected(self, isolated_db):
        rec = record_inference("img_0", "model_1", [], image_dimensions="100x100", secret=TEST_KEY)
        record_inference_event(rec)

        # Alter image_dimensions in SQLite
        conn = sqlite3.connect(isolated_db)
        cursor = conn.cursor()
        cursor.execute("UPDATE provenance_events SET image_dimensions = '999x999'")
        conn.commit()
        conn.close()

        valid, status, anomalies = verify_provenance_chain()
        assert valid is False
        assert status == "CHAIN_BROKEN"
        assert any(a["issue"] == "FIELD_MODIFICATION" and "image_dimensions" in a["detail"] for a in anomalies)


# ===========================================================================
# 4. External Integrations Correctness Tests
# ===========================================================================

class TestExternalIntegrations:
    def test_art_adapter_returns_not_assessed_not_simulated_pass(self):
        ev_list = run_art_robustness_probe("test_model", None, "pytorch")
        assert len(ev_list) > 0
        for ev in ev_list:
            assert ev.status == "NOT_ASSESSED"
            assert "simulated" not in ev.observations.lower()

    def test_intoto_adapter_honest_status(self):
        chain = InTotoEvidenceChain()
        chain.add_link("stage1", {"m": "hash1"}, {"p": "hash2"})
        ev_list = generate_intoto_evidence(chain, "audit_1")
        assert len(ev_list) > 0
        for ev in ev_list:
            assert ev.status != "PASS"
            assert "NOT an externally verified" in ev.observations


# ===========================================================================
# 5. Dataset Audit Execution Tests
# ===========================================================================

class TestDatasetAuditExecution:
    def test_label_flipping_missing_reference_is_not_assessed(self):
        cand = {"img1.jpg": [{"class_id": 0}]}
        res = check_label_flipping(cand, reference_annotations=None)
        assert res["result"] == "NOT_ASSESSED"
        assert res["check_id"] == "SEC-DS-002"

    def test_label_flipping_with_reference(self):
        cand = {"img1.jpg": [{"class_id": 1}]}
        ref = {"img1.jpg": [{"class_id": 0}]}
        res = check_label_flipping(cand, reference_annotations=ref)
        assert res["result"] == "ANOMALY_DETECTED"
        assert res["evidence"]["flip_count"] == 1

    def test_ood_samples_insufficient_samples(self):
        records = [{"filename": "img1.jpg", "pixel_stats": {"mean_r": 100, "mean_g": 100, "mean_b": 100}}]
        res = check_ood_samples(records)
        assert res["result"] == "NOT_ASSESSED"
        assert res["check_id"] == "SEC-DS-005"

    def test_ood_samples_detection(self):
        # 10 normal images + 1 outlier
        records = [
            {"filename": f"norm_{i}.jpg", "pixel_stats": {"mean_r": 120 + i, "mean_g": 120 - i, "mean_b": 120}}
            for i in range(10)
        ]
        # Outlier with extreme mean
        records.append({"filename": "outlier.jpg", "pixel_stats": {"mean_r": 10, "mean_g": 250, "mean_b": 10}})
        res = check_ood_samples(records)
        assert res["check_id"] == "SEC-DS-005"
        assert res["result"] == "ANOMALY_DETECTED"
        assert len(res["evidence"]["flagged_candidates"]) >= 1


# ===========================================================================
# 6. Model Loading Safety Tests
# ===========================================================================

class TestModelLoadingSafety:
    def test_pytorch_loading_does_not_execute_unsafe_unpickle(self, tmp_path):
        # Create an object that fails weights_only=True
        # E.g. a file containing non-tensor Python object
        import pickle
        evil_file = tmp_path / "model_evil.pt"
        class MaliciousPayload:
            def __reduce__(self):
                return (dict, ())
        with open(evil_file, "wb") as f:
            pickle.dump(MaliciousPayload(), f)

        meta = _try_load_pytorch(evil_file)
        # Should record error safely rather than executing arbitrary unpickle
        assert meta["error"] is not None
        assert "Safe loading" in meta["error"]


# ===========================================================================
# 7. API Security Tests
# ===========================================================================

class TestApiSecurity:
    def test_image_path_traversal_rejected(self):
        # Attempt to access outside ROOT
        resp = client.get("/api/image?path=../../Windows/win.ini")
        assert resp.status_code in (403, 404)

    def test_audit_id_traversal_rejected(self):
        # Path traversal inside audit_id path parameter
        resp = client.get("/api/audit/records/..%2F..%2Fetc%2Fpasswd")
        assert resp.status_code in (400, 404)

    def test_upload_dataset_zip_slip_rejected(self):
        # Create a malicious in-memory zip file containing a traversal member
        buf = BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("../../evil.txt", "malicious payload")
        buf.seek(0)

        resp = client.post(
            "/api/upload/dataset",
            files={"file": ("slip.zip", buf, "application/zip")},
            data={"format": "yolo"}
        )
        assert resp.status_code == 400
        assert "traversal" in resp.json()["detail"].lower()

    def test_upload_dataset_symlink_rejected(self):
        buf = BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zi = zipfile.ZipInfo("symlink_entry.txt")
            # Set POSIX symlink attribute (0o120000 << 16)
            zi.external_attr = 0o120777 << 16
            zf.writestr(zi, "/etc/passwd")
        buf.seek(0)

        resp = client.post(
            "/api/upload/dataset",
            files={"file": ("symlink.zip", buf, "application/zip")},
            data={"format": "yolo"}
        )
        assert resp.status_code == 400
        assert "symlink" in resp.json()["detail"].lower()

    def test_upload_dataset_decompression_bomb_rejected(self, monkeypatch):
        from src.api import app as api_app
        # Temporarily set MAX_EXTRACT_SIZE low (e.g. 1000 bytes) for unit testing
        monkeypatch.setattr(api_app, "MAX_EXTRACT_SIZE", 1000)
        buf = BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            # High compression ratio payload
            zf.writestr("huge.txt", "A" * 50000)
        buf.seek(0)

        resp = client.post(
            "/api/upload/dataset",
            files={"file": ("bomb.zip", buf, "application/zip")},
            data={"format": "yolo"}
        )
        assert resp.status_code == 400
        assert "limit" in resp.json()["detail"].lower()

