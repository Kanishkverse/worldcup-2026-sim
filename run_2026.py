"""2026 Monte Carlo (LLM tactical agent) + demo match -> CSV/JSON artifacts.

Probes the configured OpenAI endpoint first; if it is unreachable, falls
back to a local Ollama server (gemma3) by overriding the WC_LLM_* env vars
before any worker starts.
"""

import json
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("run2026")


def llm_reachable() -> bool:
    try:
        from openai import OpenAI
        c = OpenAI(base_url=os.getenv("WC_LLM_BASE_URL"),
                   api_key=os.getenv("WC_LLM_API_KEY"), timeout=20, max_retries=1)
        r = c.chat.completions.create(model=os.getenv("WC_LLM_MODEL"),
                                      messages=[{"role": "user", "content": "reply ok"}],
                                      max_tokens=5)
        return bool(r.choices)
    except Exception as exc:
        LOG.warning("OpenAI probe failed: %s", exc)
        return False


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv("en.txt")

    backend = "openai"
    if not llm_reachable():
        LOG.warning("Falling back to local Ollama (gemma3).")
        os.environ["WC_LLM_BASE_URL"] = "http://127.0.0.1:11434/v1"
        os.environ["WC_LLM_MODEL"] = os.getenv("WC_OLLAMA_MODEL", "gemma3:1b")
        os.environ["WC_LLM_API_KEY"] = "EMPTY"
        backend = "ollama"

    import worldcup_2026_sim as wc  # import AFTER overrides: module reads env at import

    # (demo match artifact is produced by redo_demo.py)

    # -- 100-iteration Monte Carlo with the LLM tactical agent ---------------
    eng = wc.MonteCarloEngine(iterations=100, workers=5, base_seed=2026, use_llm=True)
    df = eng.run()
    df.to_csv("out_2026_probs.csv", index=False)
    with open("out_2026_meta.json", "w") as fh:
        json.dump({"iterations": 100, "backend": backend,
                   "model": os.environ.get("WC_LLM_MODEL"),
                   "base_url": os.environ.get("WC_LLM_BASE_URL")}, fh, indent=2)
    print(df.head(20).to_string(index=False))
    print("saved -> out_2026_probs.csv")


if __name__ == "__main__":
    main()
