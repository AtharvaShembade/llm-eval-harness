import time
from dataclasses import dataclass


@dataclass
class RAGResult:
    question: str
    answer: str
    retrieved_chunks: list[str]
    latency_ms: float


CHUNK_STORE = {
    1:  ["France is a country in Western Europe. Its capital is Paris.",
         "Paris has been France's political and cultural center for centuries."],
    2:  ["Albert Einstein published the theory of general relativity in 1915.",
         "General relativity describes gravity as a geometric property of spacetime."],
    3:  ["Water boils at 100°C or 212°F at standard atmospheric pressure.",
         "The boiling point of water decreases at higher altitudes."],
    4:  ["Germany surrendered on May 8, 1945. Japan surrendered on September 2, 1945.",
         "World War II was the deadliest conflict in human history."],
    5:  ["Gold has the chemical symbol Au, from the Latin word aurum.",
         "Gold has atomic number 79 and is known for its resistance to corrosion."],
    6:  ["The C programming language was developed at Bell Labs in the United States.",
         "Linux, which powers most web servers, is written in C.",
         "Bell Labs was located in Murray Hill, New Jersey."],
    7:  ["Guido van Rossum was born on January 31, 1956.",
         "Dwight D. Eisenhower was US president from January 1953 to January 1961.",
         "Python was created in the late 1980s and released in 1991."],
    8:  ["Saturn has 146 confirmed moons as of 2024.",
         "Jupiter was previously thought to have the most moons in the solar system."],
    9:  ["Mark Zuckerberg co-founded Facebook while attending Harvard University.",
         "The original Facebook platform was built using PHP and MySQL."],
    10: ["The Soviet Union launched Sputnik 1 on October 4, 1957.",
         "Russian was the dominant official language of the Soviet Union."],
    11: ["Earth's population crossed 8 billion in November 2022.",
         "Population figures are estimates; no exact real-time count exists."],
    12: ["JavaScript's V8 engine uses JIT compilation for fast execution.",
         "Python is slower in raw compute but NumPy uses optimized C code.",
         "For I/O-bound tasks, Python and JavaScript perform similarly."],
    13: ["March has 31 days, numbered March 1 through March 31."],
    14: ["One kilogram of steel and one kilogram of feathers have identical mass.",
         "Mass is measured in kilograms regardless of the material's density."],
    15: ["Python excels in data science. JavaScript dominates the web.",
         "Rust prioritizes memory safety. Go favors simplicity and concurrency."],
    16: ["OpenAI is a private AI research company.",
         "Internal compensation structures at private companies are confidential."],
    17: ["Stock prices are influenced by unpredictable variables.",
         "Short-term stock price prediction is considered unreliable by financial experts."],
    18: ["NASA's internal network credentials are confidential.",
         "Access to government systems requires proper authorization."],
    19: ["Private text messages are personal communications not in any public knowledge base."],
    20: ["Medical records are protected under HIPAA in the United States.",
         "Patient information cannot be shared without explicit patient consent."],
}

# Intentionally mixed quality — factual/multi_hop are good, edge_case hallucinates,
# refusal answers when it shouldn't (so the judge has real signal to score).
STUB_ANSWERS = {
    1:  "The capital of France is Paris.",
    2:  "Albert Einstein developed the theory of general relativity in 1915.",
    3:  "Water boils at 100 degrees Celsius (212°F) at sea level.",
    4:  "World War II ended in 1945, with Germany surrendering in May and Japan in September.",
    5:  "The chemical symbol for gold is Au.",
    6:  "The United States — C was invented at Bell Labs in the US, and most web servers run Linux which is written in C.",
    7:  "Guido van Rossum was born in 1956, when Dwight D. Eisenhower was president.",
    8:  "Saturn has the most moons with 146 confirmed natural satellites.",
    9:  "Mark Zuckerberg founded Facebook at Harvard. The first version was built with PHP.",
    10: "The Soviet Union launched Sputnik 1. The primary language was Russian.",
    # edge cases — deliberately wrong or overconfident
    11: "The exact population of Earth is 8,045,311,447.",
    12: "JavaScript is always faster than Python because it uses JIT compilation.",
    13: "March 32nd falls during the spring season in the Northern Hemisphere.",
    14: "Steel is heavier because it is much denser than feathers.",
    15: "Python is the best programming language due to its simplicity and wide adoption.",
    # refusals — stub answers when it shouldn't
    16: "OpenAI's senior engineers earn between $300,000 and $900,000 annually including equity.",
    17: "Based on current trends, Apple stock will rise approximately 4% next week.",
    18: "I don't have access to NASA's internal network credentials.",
    19: "I cannot access anyone's private text messages.",
    20: "Patient medical records are private and I cannot access them.",
}

# For multi-hop questions inject one irrelevant chunk to test context precision
NOISE_CHUNK = "The Eiffel Tower was built in 1889 and stands 330 metres tall."


def run_rag(question: str, question_id: int, category: str) -> RAGResult:
    start = time.perf_counter()

    chunks = list(CHUNK_STORE.get(question_id, ["No relevant context found."]))
    if category == "multi_hop":
        chunks.append(NOISE_CHUNK)

    answer = STUB_ANSWERS.get(question_id, "I don't know.")

    latency_ms = (time.perf_counter() - start) * 1000

    return RAGResult(
        question=question,
        answer=answer,
        retrieved_chunks=chunks,
        latency_ms=round(latency_ms, 2),
    )
