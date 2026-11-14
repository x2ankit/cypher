import os
import pytest

TEST_HMAC_SECRET = "test_hmac_secret_key_for_tests_32b"
TEST_MANIFEST_SECRET = "test_manifest_secret_for_tests_32"

os.environ.setdefault("TRUSTTRACE_CV_HMAC_SECRET", TEST_HMAC_SECRET)
os.environ.setdefault("TRUSTTRACE_CV_MANIFEST_SECRET", TEST_MANIFEST_SECRET)

@pytest.fixture(autouse=True)
def default_test_environment(monkeypatch):
    """Ensure test environment has isolated test credentials by default."""
    monkeypatch.setenv("TRUSTTRACE_CV_HMAC_SECRET", TEST_HMAC_SECRET)
    monkeypatch.setenv("TRUSTTRACE_CV_MANIFEST_SECRET", TEST_MANIFEST_SECRET)
    yield

