"""Compatibility entry point: ``uvicorn backend.main_api:app``."""

from __future__ import annotations

import logging
import os

from .api import app


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    uvicorn.run(
        "backend.main_api:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
