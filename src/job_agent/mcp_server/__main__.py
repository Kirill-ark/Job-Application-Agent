"""Run the MCP server over stdio: `python -m job_agent.mcp_server`.

Settings come from the environment (.env):
  CV_PATH        the CV to serve (default: data/cv.yaml, the example CV)
  DATABASE_PATH  the SQLite file (default: data/job_agent.db)
  GROQ_API_KEY   without it, scores get template reasons and letters are unavailable
  GROQ_LETTER_MODEL  model for letters (default: openai/gpt-oss-120b)
  REASONS_WITH_LLM   "0" for template reasons only; the agent sets it to save tokens

stdout carries the MCP protocol, so logs go to stderr only.
"""

import logging
import os

from dotenv import load_dotenv

from job_agent import REPO_ROOT
from job_agent.cv import DEFAULT_CV_PATH
from job_agent.db import default_db_path
from job_agent.letters import LETTER_SCHEMA
from job_agent.llm import DEFAULT_LETTER_MODEL, LLMError, make_complete
from job_agent.mcp_server.server import build_server

logger = logging.getLogger("job_agent.mcp_server")


def main() -> None:
    logging.basicConfig(level=logging.INFO)  # basicConfig writes to stderr
    load_dotenv(REPO_ROOT / ".env")
    # Relative paths are relative to the repo, like DATABASE_PATH: MCP clients
    # such as Claude Desktop start the server from another folder.
    cv_path = REPO_ROOT / os.environ["CV_PATH"] if os.environ.get("CV_PATH") else DEFAULT_CV_PATH
    try:
        letter_model = os.environ.get("GROQ_LETTER_MODEL") or DEFAULT_LETTER_MODEL
        letter_llm = make_complete(2500, LETTER_SCHEMA, model=letter_model)
        reasons_llm = make_complete() if os.environ.get("REASONS_WITH_LLM") != "0" else None
    except LLMError as error:
        logger.warning("No LLM (%s): template reasons only, no letters", error)
        reasons_llm = letter_llm = None
    server = build_server(default_db_path(), cv_path, reasons_llm, letter_llm)
    server.run("stdio")


if __name__ == "__main__":
    main()
