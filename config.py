"""Project configuration helpers.

Loads the project-local .env file without overwriting variables already set in
 the shell. Priority is therefore: CLI arguments > shell environment > .env >
 application defaults.
"""
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
ENV_FILE = PROJECT_DIR / ".env"


def load_project_env(env_path=None, *, override=False):
    """Load the project's .env file (or an explicit path) into os.environ."""
    path = Path(env_path).expanduser() if env_path else ENV_FILE
    return load_dotenv(dotenv_path=path, override=override)
