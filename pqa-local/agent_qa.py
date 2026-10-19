"""Optional: exercise paper-qa's agent path (tool-calling) with DeepSeek."""

import asyncio
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from paperqa.agents import ask
from qa import build_settings


async def main() -> None:
    settings = build_settings()
    settings.agent.index.paper_directory = str(BASE_DIR / "papers")
    question = (
        sys.argv[1] if len(sys.argv) > 1 else "What is this paper about?"
    )
    print(f"Agent query: {question}", flush=True)
    t0 = time.perf_counter()
    response = await ask(question, settings=settings)
    print(f"Answered in {time.perf_counter() - t0:.1f}s", flush=True)
    session = response.session
    print("\n=== Answer ===")
    print(session.answer)
    print(f"\n=== Contexts used: {len(session.contexts)} ===")


if __name__ == "__main__":
    asyncio.run(main())
