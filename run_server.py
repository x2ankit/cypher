"""
run_server.py

Start the CYPHER FastAPI server.

Usage:
  python run_server.py [--host HOST] [--port PORT] [--reload]

Default: http://127.0.0.1:8000
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
sys.path.insert(0, ".")

import uvicorn

if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )
