from __future__ import annotations

import os

import uvicorn

from .app import create_app


def run() -> None:
    host = os.getenv("SHOWHOW_RECORDER_HOST", "127.0.0.1")
    port = int(os.getenv("SHOWHOW_RECORDER_PORT", "18080"))
    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run()
