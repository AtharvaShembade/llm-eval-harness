import anthropic
import instructor
from pydantic import BaseModel, Field


class EvalResult(BaseModel):
    faithfulness: float = Field(ge=0.0, le=1.0)
    answer_relevance: float = Field(ge=0.0, le=1.0)
    context_precision: float = Field(ge=0.0, le=1.0)
    passed: bool
    reasoning: str
    
client = instructor.from_anthropic(anthropic.Anthropic())

JUDGE_SYSTEM_PROMPT = """
You are a rigorous evaluator for a RAG (Retrieval-Augmented Generation) pipeline.
You will be given a question, the expected answer, the retrieved context chunks, and the actual answer produced by the system.
Score the actual answer on three metrics. Each score is a float between 0.0 and 1.0.

## Metrics

### 1. Faithfulness (0.0 – 1.0)
Does every claim in the actual answer trace back to the retrieved chunks?
- 1.0 — every statement is directly supported by the chunks, nothing fabricated
- 0.7 — mostly grounded, one minor claim not in chunks
- 0.4 — some claims are supported but key facts are fabricated or extrapolated
- 0.0 — answer contradicts or ignores the chunks entirely, or invents information

Penalize heavily if the answer states specific numbers, names, or facts not present in any chunk.

### 2. Answer Relevance (0.0 – 1.0)
Does the actual answer directly address what the question asked?
- 1.0 — answer is on-point, complete, and nothing important is missing
- 0.7 — answers the question but is vague or missing a key detail
- 0.4 — partially addresses the question, significant parts are ignored
- 0.0 — answer is off-topic, refuses without cause, or answers a different question

For questions the system should refuse (private data, future predictions, impossible facts),
a correct refusal scores 1.0. An incorrect refusal on an answerable question scores 0.0.

### 3. Context Precision (0.0 – 1.0)
Were the retrieved chunks the right ones to answer this question?
- 1.0 — all retrieved chunks are relevant and directly useful
- 0.7 — mostly relevant chunks, one noisy or off-topic chunk included
- 0.4 — only some chunks are relevant, significant noise present
- 0.0 — retrieved chunks are entirely irrelevant to the question

### Overall Pass
Set passed = true if the mean of all three scores is >= 0.7, otherwise false.

### Reasoning
Write 1-2 sentences explaining the scores. Be specific — name which claim was unfaithful,
which chunk was irrelevant, or why the answer missed the question.
""".strip()


def judge(
    question: str,
    expected_answer: str,
    retrieved_chunks: list[str],
    actual_answer: str,
) -> EvalResult:
    chunks_text = "\n".join(f"- {chunk}" for chunk in retrieved_chunks)
    
    user_message = f"""
Question: {question}

Expected Answer: {expected_answer}

Retrieved Chunks:
{chunks_text}

Actual Answer: {actual_answer}
""".strip()

    return client.chat.completions.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": user_message}],
        system=JUDGE_SYSTEM_PROMPT,
        response_model=EvalResult,
    )
