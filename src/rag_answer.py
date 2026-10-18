"""The result contract shared by the naive baseline, the agent and the evals.

Both systems return a RAGAnswer, so Phase 4 can compare them without knowing
which one produced it. The LLM settings live here too: baseline and agent must
use the same model, and one definition keeps them from drifting apart.
"""

from dataclasses import dataclass, field

MODEL = "claude-haiku-5-5"
EFFORT = "low"
# Rate card for prompts up to 100K tokens ($0.50 / $2.50 above); ours are a
# few thousand. Cache writes cost 1.25x input (5-minute TTL), reads 0.1x.
PRICE_PER_MTOK = {"input": 0.10, "output": 0.50, "cache_write": 0.125, "cache_read": 0.01}


@dataclass
class RAGAnswer:
    question: str
    answer: str
    retrieved: list[dict]               # chunk_id, doc_id, score, ... in retrieval order
    input_tokens: int                   # uncached input
    output_tokens: int                  # includes thinking
    latency_s: float
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    llm_calls: int = 1
    meta: dict = field(default_factory=dict)

    @property
    def cost_usd(self) -> float:
        return (self.input_tokens * PRICE_PER_MTOK["input"]
                + self.output_tokens * PRICE_PER_MTOK["output"]
                + self.cache_write_tokens * PRICE_PER_MTOK["cache_write"]
                + self.cache_read_tokens * PRICE_PER_MTOK["cache_read"]) / 1_000_000

    @property
    def retrieved_doc_ids(self) -> list[str]:
        """Distinct documents in rank order (what Phase 4 checks against)."""
        return list(dict.fromkeys(h["doc_id"] for h in self.retrieved))
