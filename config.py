"""Shared project configuration loading.

Loads the project-local .env file before modules read environment variables.
Existing shell environment variables are preserved.
"""
from __future__ import annotations

from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional until dependencies are installed
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"

if load_dotenv is not None:
    load_dotenv(dotenv_path=ENV_FILE, override=False)
