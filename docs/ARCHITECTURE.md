# Architecture

CYPHER is designed as an offline-first, modular Python application. The architecture prioritizes separation of concerns, ensuring that dataset inspection, model integrity validation, model performance evaluation, and inference provenance are handled by independent sub-modules that feed into a unified reporting pipeline.

## System Components

```
CYPHER
├── src/
│   ├── dataset/          # Dataset integrity and label quality checks (COCO, YOLO)
│   │   ├── inspector.py
│   │   └── security_checks.py
│   ├── models/           # Model artifact validation, performance, hashing, and behavior
│   │   ├── integrity.py
│   │   ├── performance.py
│   │   ├── behavioral_fingerprint.py
│   │   ├── trigger_probe.py
│   │   ├── controlled_transformations.py
│   │   └── runtime.py
│   ├── inference/        # Cryptographic signing, provenance chain, and tamper detection
│   │   ├── verifier.py
│   │   └── provenance.py
│   ├── evidence/         # Evidence correlation and unified verdict resolution
│   │   └── correlator.py
│   ├── reference/        # Reference profiles and dataset baseline statistics
│   │   └── profile.py
│   ├── shift/            # Feature and distribution shift tracking
│   │   └── distribution.py
│   ├── reporting/        # Report generation, HTML export, and coverage documentation
│   │   ├── html_generator.py
│   │   ├── benchmark_runner.py
│   │   └── report_generator.py
│   ├── integrations/     # Adapters for upstream open-source tools
│   │   ├── art_adapter.py
│   │   ├── backdoorbench_adapter.py
│   │   ├── cleanlab_adapter.py
│   │   ├── cleanvision_adapter.py
│   │   ├── cosign_adapter.py
│   │   ├── intoto_adapter.py
│   │   └── trojai_adapter.py
│   ├── api/              # FastAPI backend and tamper-evident SQLite audit ledger
│   │   ├── app.py
│   │   └── audit_db.py
│   ├── ui/               # HTML/CSS/JS Dynamic frontend
│   │   ├── index.html
│   │   ├── style.css
│   │   └── app.js
│   └── cli.py            # Typer-based command-line interface
├── scripts/              # Independent execution and generation scripts
│   ├── generate_fixtures.py
│   ├── generate_benchmark.py
│   ├── run_benchmark.py
│   └── demo_behavior.py
├── tests/                # 171 automated pytest cases (cryptographic integrity, provenance, dataset benchmarking, security regression)
└── docs/                 # Comprehensive documentation
```

## Data Flow

1. **Dataset Audit:** User provides dataset path -> `dataset/inspector.py` extracts metadata -> `dataset/security_checks.py` runs statistical checks (flooding, flipping, triggers, poisoning, OOD) -> `CleanVision` adapter runs -> Complete findings array generated.
2. **Model Audit:** User provides artifact + manifest -> `models/integrity.py` structurally validates format safely (`weights_only=True` for PyTorch) and verifies cryptographic signatures (HMAC or RSA) -> Integration adapters run -> Findings array generated.
3. **Inference Audit:** User provides `.jsonl` inference log -> `inference/verifier.py` validates canonical hashes, HMAC-SHA256 signatures, sequence monotonicity, and replays -> Findings array generated.
4. **Full Audit:** Integrates all the above, computes `Cleanlab` label issues, calculates CV Performance (AP/mAP, IoU) and Behavioral Fingerprints, then constructs local provenance metadata.
5. **Report Generation:** `reporting/report_generator.py` aggregates all findings, determines an explicit verdict (`PASS`, `ANOMALIES_DETECTED`, `FAIL`, `INCONCLUSIVE`, `NOT_ASSESSED`), calculates evaluation coverage, and emits structured JSON/HTML reports.

## Upstream Integrations

To comply with the offline requirement and eliminate false security claims, upstream repositories are integrated via explicit Adapters (`src/integrations`):
- **CleanVision** and **Cleanlab**: Execute genuine checks if dependencies and datasets are present locally; emit `NOT_ASSESSED` when libraries or required references are absent.
- **IBM ART**, **BackdoorBench**, and **TrojAI**: Structural adapters that emit honest `NOT_ASSESSED` findings in default offline environments rather than simulated passes.
- **in-toto** and **Cosign**: Generate verifiable local provenance structures; external attestations/signatures return `NOT_ASSESSED` when CLI/certificate infrastructure is unconfigured.
