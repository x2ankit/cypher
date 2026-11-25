# Threat Model

TRUSTTRACE CV addresses specific threats in a multi-contributor Computer Vision pipeline. This threat model bounds the scope of the prototype's capabilities.

## Addressed Threats

1. **Data Poisoning & Label Flipping (Dataset level)**
   - **Threat:** A contributor maliciously alters training labels or injects statistically anomalous images to degrade model performance.
   - **Mitigation:** Statistical screening (Z-score pixel distributions, bounding box label variance) implemented in `SEC-DS-002` and `SEC-DS-004`.

2. **Trigger Patterns / Backdoors (Dataset level)**
   - **Threat:** A contributor injects visible, highly structured patches (e.g., solid squares) into training images as a backdoor trigger.
   - **Mitigation:** Shannon entropy calculation over sliding windows. Solid patches exhibit near-zero entropy compared to natural image variance. Implemented in `SEC-DS-003`.

3. **Duplicate Flooding (Dataset level)**
   - **Threat:** A contributor floods the dataset with duplicate files to manipulate batch statistics or skew training.
   - **Mitigation:** Cryptographic hashing (SHA-256) of every image file. Identifies exact duplicates (`SEC-DS-001`).

4. **Model Substitution (Model level)**
   - **Threat:** An approved model artifact is silently replaced with a malicious or compromised version.
   - **Mitigation:** Strict cryptographic digest verification against an authorized manifest file (`SEC-MDL-001`).

5. **Inference Tampering & Replay (Inference level)**
   - **Threat:** Inference logs are altered post-prediction to hide false positives, or valid logs are replayed to simulate activity.
   - **Mitigation:** Canonical JSON serialization and HMAC-SHA256 signatures over the input/model/prediction bundle (`SEC-INF-001`). Nonces and deduplication for replay attacks (`SEC-INF-002`).

## Out of Scope (Known Limitations)

1. **Invisible or Perceptual Triggers:** Sophisticated backdoors utilizing adversarial noise imperceptible to the human eye cannot be detected by basic entropy scans.
2. **Model Weight Scanning:** We do not execute universal backdoor detection via static weight analysis, as this is currently an unsolved mathematical problem for generalized deep neural networks without access to the exact training set.
3. **Extreme Geometric Warping:** While perceptual difference hashing (dHash, `SEC-DS-001b`) detects recompressed (JPEG quality shift) and resized duplicates, severe non-rigid geometric transformations (large affine rotations or extreme perspective skew) require heavier feature-invariant descriptors that exceed lightweight heuristic inspection.
4. **Hardware-Enforced Immutability:** The local SQLite audit log provides tamper-evidence via cryptographic hash chaining and sequence enforcement; it does not replace a physically write-once (WORM) hardware storage system.

## Assurance Verdict Framework

Audit reports and API endpoints adhere to five explicit verdict states:
- **`PASS`**: All required, applicable integrity and security checks ran and passed with zero anomalies.
- **`ANOMALIES_DETECTED`**: Executed checks successfully identified heuristic anomalies (e.g., duplicate flooding, entropy drop, label flips, OOD outliers).
- **`FAIL`**: Definite security verification failure (e.g., invalid cryptographic signature, hash mismatch, provenance sequence break, payload tampering).
- **`INCONCLUSIVE`**: Findings were empty or evidence was insufficient to establish an assurance result. Empty results never default to `PASS`.
- **`NOT_ASSESSED`**: A check could not execute because an optional dependency, reference baseline, or signing credential was absent.

## Cryptographic Key Management & Integrity Guarantees

1. **Zero Hardcoded Secrets**: All fallback signing secrets have been removed. Keys must be explicitly configured via `TRUSTTRACE_CV_HMAC_SECRET` and `TRUSTTRACE_CV_MANIFEST_SECRET`. Missing keys yield explicit failures or `NOT_ASSESSED` verdicts.
2. **Canonical Serialization**: Manifests and inference payloads are deterministically canonicalized (`sort_keys=True`, `separators=(',', ':')`) prior to digest computation and signing, eliminating JSON spacing discrepancies.
3. **Timing-Attack Resistance**: All HMAC signature verifications strictly employ `hmac.compare_digest`.
4. **Provenance Chain Verification**: Provenance records are ordered by actual append sequence (`rowid`), verifying sequence continuity (`sequence_number`), hash chaining (`previous_record_hash`), and payload integrity.

## Safe Model Loading & Inspection

1. **Restricted Deserialization**: PyTorch model inspection uses `weights_only=True` to prevent arbitrary code execution during inspection of untrusted model checkpoints.
2. **Structural Isolation**: Model architecture inspection does not invoke forward passes or arbitrary unpickling on unverified checkpoints.

## API & Filesystem Defenses

1. **Path Traversal Defenses**: Dataset ZIP extraction validates that all target file paths resolve safely within the intended extraction directory (Zip Slip mitigation).
2. **Resource Boundaries**: Strict extraction quotas are enforced (max 50 MB archive size, max 100 MB decompressed footprint, max 1,000 files).
3. **Identifier Sanitization**: Audit IDs are strictly validated (`^[a-zA-Z0-9_\-]{1,64}$`) preventing directory traversal in audit endpoints.
4. **Safe File Resolution**: Endpoint path parameters are constrained within the project root directory.
