import os
import pytest

TEST_HMAC_SECRET = "test_hmac_secret_key_for_tests_32b"
TEST_MANIFEST_SECRET = "test_manifest_secret_for_tests_32"

os.environ.setdefault("TRUSTTRACE_CV_HMAC_SECRET", TEST_HMAC_SECRET)
os.environ.setdefault("TRUSTTRACE_CV_MANIFEST_SECRET", TEST_MANIFEST_SECRET)

@pytest.fixture(autouse=True)
def default_test_environment(monkeypatch, tmp_path):
    """Ensure test environment has isolated test credentials and temporary DBs by default."""
    monkeypatch.setenv("TRUSTTRACE_CV_HMAC_SECRET", TEST_HMAC_SECRET)
    monkeypatch.setenv("TRUSTTRACE_CV_MANIFEST_SECRET", TEST_MANIFEST_SECRET)
    monkeypatch.setenv("TRUSTTRACE_AUDIT_DB", str(tmp_path / "test_audit_events.sqlite"))
    monkeypatch.setenv("TRUSTTRACE_PROVENANCE_DB", str(tmp_path / "test_provenance.db"))
    yield

