"""Run the web app: python -m job_agent.api, then open http://localhost:8000."""

import logging
import os

import uvicorn
from dotenv import load_dotenv

from job_agent.api.app import create_app
from job_agent.db import REPO_ROOT


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    load_dotenv(REPO_ROOT / ".env")
    uvicorn.run(
        create_app(trusted_proxies=int(os.environ.get("TRUST_PROXY_HEADERS") or 0)),
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
    )


if __name__ == "__main__":
    main()
