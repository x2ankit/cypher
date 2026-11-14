"""
src/inference/provenance.py

Inference provenance tracking and tamper-evident event logging in SQLite.

Limitation & Trust Model:
  SQLite provides local tamper-evident event ordering and hash chaining.
  Without an external immutable anchor (such as a transparency log or RFC 3161
  timestamp authority), a privileged local database administrator with direct disk
  access can recalculate the hash chain. SQLite is NOT hardware-immutable.
  Time freshness expiration is unsupported by design to permit retroactive offline audits.
"""

import sqlite3
import json
import hashlib
import time
import os
import uuid
from typing import Dict, Any, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)

def get_db_path() -> str:
    return os.environ.get("TRUSTTRACE_PROVENANCE_DB", "data/provenance.db")

def init_db():
    db_path = get_db_path()
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS provenance_events (
            event_id TEXT PRIMARY KEY,
            input_sha256 TEXT,
            image_dimensions TEXT,
            model_id TEXT,
            model_sha256 TEXT,
            verified_manifest_id TEXT,
            preprocessing_config TEXT,
            preprocessing_digest TEXT,
            inference_config TEXT,
            prediction_payload TEXT,
            output_digest TEXT,
            timestamp REAL,
            nonce TEXT,
            sequence_number INTEGER,
            previous_event_hash TEXT,
            record_hash TEXT UNIQUE,
            verification_status TEXT,
            canonical_payload TEXT
        )
    ''')
    try:
        cursor.execute("ALTER TABLE provenance_events ADD COLUMN canonical_payload TEXT")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()

def canonicalize_record(record: Dict[str, Any]) -> str:
    """Canonicalize the record before hashing/signing."""
    return json.dumps(record, sort_keys=True, separators=(',', ':'))

def record_inference_event(event_data: Dict[str, Any]) -> Tuple[bool, str, str]:
    """
    Records an inference event in SQLite tamper-evident store.
    Checks for tampering, sequence validation, and replay using record_hash.
    Returns (success, message, record_hash).
    """
    init_db()
    
    # Map old format to new format
    if "event_id" not in event_data and "record_id" in event_data:
        event_data["event_id"] = event_data["record_id"]
    if "event_id" not in event_data:
        event_data["event_id"] = str(uuid.uuid4())

    event_id = str(event_data.get("event_id", "")).strip()
    if not event_id:
        return False, "INVALID_EVENT_ID", ""
        
    if "timestamp" not in event_data:
        event_data["timestamp"] = time.time()
        
    if "nonce" not in event_data:
        event_data["nonce"] = os.urandom(16).hex()

    seq_num = event_data.get("sequence_number", 0)
    if not isinstance(seq_num, int) or seq_num < 0:
        return False, "INVALID_SEQUENCE_NUMBER", ""

    from src.inference.verifier import canonicalize_inference_payload
    import hmac

    if "payload_hash" in event_data:
        canonical_bytes = canonicalize_inference_payload(event_data)
        expected_hash = hashlib.sha256(canonical_bytes).hexdigest()
        if not hmac.compare_digest(str(event_data["payload_hash"]), expected_hash):
            return False, "PAYLOAD_HASH_MISMATCH", ""
        record_hash = event_data["payload_hash"]
        canonical_str = canonical_bytes.decode('utf-8')
    else:
        canonical_str = canonicalize_record(event_data)
        record_hash = hashlib.sha256(canonical_str.encode('utf-8')).hexdigest()

    db_path = get_db_path()
    try:
        conn = sqlite3.connect(db_path, timeout=30.0)
        cursor = conn.cursor()
        cursor.execute("BEGIN IMMEDIATE")

        # Replay defense: verify nonce uniqueness if nonce is provided
        nonce_val = event_data.get("nonce")
        if nonce_val:
            cursor.execute("SELECT 1 FROM provenance_events WHERE nonce = ?", (nonce_val,))
            if cursor.fetchone():
                conn.rollback()
                conn.close()
                return False, "REPLAY_DETECTED", record_hash
        
        cursor.execute('''
            INSERT INTO provenance_events (
                event_id, input_sha256, image_dimensions, model_id, model_sha256, 
                verified_manifest_id, preprocessing_config, preprocessing_digest, 
                inference_config, prediction_payload, output_digest, timestamp, 
                nonce, sequence_number, previous_event_hash, record_hash, verification_status,
                canonical_payload
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            event_data.get("event_id"),
            event_data.get("input_sha256"),
            event_data.get("image_dimensions"),
            event_data.get("model_id"),
            event_data.get("model_sha256"),
            event_data.get("verified_manifest_id"),
            json.dumps(event_data.get("preprocessing_config", {})),
            event_data.get("preprocessing_digest"),
            json.dumps(event_data.get("inference_config", {})),
            json.dumps(event_data.get("prediction_payload", event_data.get("predictions", []))),
            event_data.get("output_digest"),
            event_data.get("timestamp"),
            event_data.get("nonce"),
            seq_num,
            event_data.get("previous_event_hash"),
            record_hash,
            "ACCEPTED",
            canonical_str
        ))
        conn.commit()
        conn.close()
        return True, "Event recorded successfully", record_hash
    except sqlite3.IntegrityError:
        # Replay detected (record_hash or event_id already exists)
        conn.close()
        return False, "REPLAY_DETECTED", record_hash
    except Exception as e:
        conn.close()
        logger.error(f"Error recording provenance event: {e}")
        return False, str(e), record_hash

def verify_provenance_chain() -> Tuple[bool, str, List[Dict[str, Any]]]:
    """
    Verifies the integrity of the append-only provenance event trail.
    Detects modification of signed fields, deletion, reordering, and broken hash links.
    """
    init_db()
    db_path = get_db_path()
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Order by append/insertion order (rowid) to verify chain continuity and sequence progression
        cursor.execute("SELECT rowid, * FROM provenance_events ORDER BY rowid ASC")
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return True, "Chain is empty (valid).", []
            
        anomalies = []
        last_hash = None
        prev_seq = None
        import hmac

        for i, row in enumerate(rows):
            current_hash = row["record_hash"]
            prev_hash_claimed = row["previous_event_hash"]
            seq = row["sequence_number"]
            event_id = row["event_id"]

            # Validate sequence number
            if not isinstance(seq, int) or seq < 0:
                anomalies.append({
                    "event_id": event_id,
                    "issue": "INVALID_SEQUENCE",
                    "detail": f"Sequence number {seq} is invalid."
                })
            elif prev_seq is not None:
                if seq < prev_seq:
                    anomalies.append({
                        "event_id": event_id,
                        "issue": "SEQUENCE_OUT_OF_ORDER",
                        "detail": f"Sequence number {seq} decreased from previous sequence {prev_seq}."
                    })
                elif seq == prev_seq:
                    anomalies.append({
                        "event_id": event_id,
                        "issue": "DUPLICATE_SEQUENCE",
                        "detail": f"Duplicate sequence number {seq} detected."
                    })
            prev_seq = seq

            # If not the first record, verify linkage
            if i > 0:
                if prev_hash_claimed != last_hash:
                    anomalies.append({
                        "event_id": event_id,
                        "issue": "BROKEN_LINK",
                        "detail": f"Claimed previous hash {prev_hash_claimed} does not match actual previous hash {last_hash}"
                    })
            
            # Verify stored record integrity against record_hash
            canonical_payload = row["canonical_payload"] if "canonical_payload" in row.keys() else None
            if canonical_payload:
                computed_hash = hashlib.sha256(canonical_payload.encode('utf-8')).hexdigest()
                if not hmac.compare_digest(current_hash or "", computed_hash):
                    anomalies.append({
                        "event_id": event_id,
                        "issue": "PAYLOAD_TAMPERED",
                        "detail": f"Stored canonical payload hash {computed_hash} does not match record_hash {current_hash}."
                    })
                else:
                    # Also verify stored column values match canonical payload
                    try:
                        parsed = json.loads(canonical_payload)
                        protected_cols = (
                            "model_id", "input_sha256", "image_dimensions", "model_sha256",
                            "verified_manifest_id", "preprocessing_digest", "output_digest",
                            "nonce", "previous_event_hash"
                        )
                        for col in protected_cols:
                            if col in parsed and col in row.keys():
                                if parsed[col] != row[col]:
                                    anomalies.append({
                                        "event_id": event_id,
                                        "issue": "FIELD_MODIFICATION",
                                        "detail": f"Column {col} '{row[col]}' altered vs canonical '{parsed[col]}'."
                                    })
                    except Exception:
                        pass
            
            last_hash = current_hash
            
        if anomalies:
            return False, "CHAIN_BROKEN", anomalies
            
        return True, "CHAIN_VALID", []
        
    except Exception as e:
        logger.error(f"Error verifying chain: {e}")
        return False, f"ERROR: {str(e)}", []
