"""
Backward-compatible entrypoint for Render / local dev.

Run:
  uvicorn main:app --host 0.0.0.0 --port 8000

Equivalent:
  uvicorn deploy.main:app --host 0.0.0.0 --port 8000
"""
import os

from deploy.main import app

__all__ = ["app"]

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
