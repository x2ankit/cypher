# API Reference

The TRUSTTRACE CV backend exposes the following REST endpoints. The API is hosted at `http://127.0.0.1:8000` by default (configurable via `HOST` and `PORT`).

## Dataset Uploads

`POST /api/upload/dataset`  
Uploads a ZIP file containing the dataset (images and annotations) with Zip-Slip protection.
- **Form Data**: `file` (binary zip archive), `format` (`yolo` or `coco`).
- **Response**: `{"dataset_path": "...", "format": "...", "message": "..."}`

## Security Check Inventory & Coverage

`GET /api/coverage`  
Returns the full inventory of implemented security checks, categories, verification status, and test coverage metadata.
- **Response**: `{"total_checks": N, "coverage_pct": ..., "checks": [...]}`

## Security & Provenance Audits

`POST /api/audit/dataset`  
Initiates an asynchronous background integrity audit on an uploaded dataset.
- **Body** (JSON): `{"dataset_path": "...", "format": "coco|yolo", "seed": 42, "audit_id": "optional-uuid"}`
- **Response**: `{"status": "started", "audit_id": "...", "task_id": "..."}`

`POST /api/audit/model`  
Executes a synchronous structural and cryptographic audit on a model artifact.
- **Body** (JSON): `{"model_path": "...", "manifest_path": "optional/manifest.json", "seed": 42}`
- **Response**: `{"report_id": "...", "verdict": "...", "severity_summary": {...}, "findings": [...]}`

`POST /api/audit/inference`  
Audits stored inference records for HMAC signature tampering, stale replaying, and anomalous backdoor behavior.
- **Body** (JSON): `{"log_path": "...", "seed": 42}`
- **Response**: `{"report_id": "...", "verdict": "...", "severity_summary": {...}, "findings": [...]}`

`POST /api/audit/full`  
Executes a combined audit across dataset, model, and inference artifacts with unified in-toto evidence linking.
- **Body** (JSON): `{"dataset_path": "...", "model_path": "...", "inference_log": "..."}`
- **Response**: `{"report_id": "...", "verdict": "...", "findings": [...]}`

## Status & Cryptographic Verification

`GET /api/audit/status`  
`GET /api/audit/status?audit_id={audit_id}`  
Returns real-time task lifecycle status (`PENDING`, `RUNNING`, `COMPLETED`, `FAILED`) and active background progress.

`GET /api/audit/verify-chain/{audit_id}`  
Cryptographically verifies the SHA-256 hash chain and sequence monotonicity of audit events in SQLite.
- **Response**: `{"is_valid": true|false, "audit_id": "...", "total_events": N, "anomalies": [], "last_hash": "..."}`

`POST /api/provenance/verify-chain`  
Verifies the SQLite inference provenance chain and sequence continuity.
- **Response**: `{"status": "valid|invalid", "message": "...", "anomalies": []}`

`GET /api/audit/events/{audit_id}`  
Returns the chronological execution history and diagnostic telemetry for an audit.
- **Response**: `{"events": [...]}`

`GET /api/audit/records/{audit_id}`  
Returns the image-by-image inspection records and perceptual hashes.

## Report Generation

`GET /api/reports/html/{audit_id}`  
Downloads a fully self-contained offline HTML forensic report with embedded base64 images and complete execution traces. Returns HTTP 404 if the audit ID does not exist.

`GET /api/reports`  
Lists all historically generated JSON reports.

## Images

`GET /api/image?path={path}`  
Serves an image file relative to the project root with path traversal validation.

`GET /api/image/find?dataset_path={dir}&filename={name}`  
Recursively locates and serves a specific filename within an extracted dataset directory.
