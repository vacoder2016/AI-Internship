"""Load .env from week-1, then fall back to the course root."""

import os
from pathlib import Path

import certifi
from dotenv import load_dotenv

_WEEK1_DIR = Path(__file__).resolve().parent
_COURSE_ROOT = _WEEK1_DIR.parents[2]


def load_course_env() -> None:
    # Windows Python installs often need an explicit CA bundle for HTTPS APIs.
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

    for env_path in (_WEEK1_DIR / ".env", _COURSE_ROOT / ".env"):
        if env_path.exists():
            load_dotenv(env_path)
            return


def make_openai_client():
    import ssl

    import httpx
    import truststore
    from openai import OpenAI

    ssl_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    return OpenAI(http_client=httpx.Client(verify=ssl_context))
