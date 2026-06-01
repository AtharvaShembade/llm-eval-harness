import argparse
import json
import subprocess

from judge import judge
from rag_runner import run_rag
from report import print_report
from store import create_run, get_last_run, init_db, save_result


def get_git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def cmd_run(notes: str | None):
    init_db()

    with open("datasets/questions.json") as f:
        questions = json.load(f)

    git_sha = get_git_sha()
    run_id = create_run(git_sha=git_sha, notes=notes)
    print(f"Run #{run_id} started (sha={git_sha}, notes={notes})\n")

    for q in questions:
        print(f"  Q{q['id']:02d} [{q['category']}] {q['question'][:60]}...")
        rag = run_rag(q["question"], q["id"], q["category"])
        result = judge(
            question=q["question"],
            expected_answer=q["expected_answer"],
            retrieved_chunks=rag.retrieved_chunks,
            actual_answer=rag.answer,
        )
        save_result(
            run_id=run_id,
            question=q["question"],
            answer=rag.answer,
            faithfulness=result.faithfulness,
            answer_relevance=result.answer_relevance,
            context_precision=result.context_precision,
            passed=result.passed,
            reasoning=result.reasoning,
        )
        status = "PASS" if result.passed else "FAIL"
        print(f"       → {status} | F={result.faithfulness:.2f} AR={result.answer_relevance:.2f} CP={result.context_precision:.2f}")

    print()
    prev_run_id = get_last_run(before_run_id=run_id)
    print_report(run_id, compare_run_id=prev_run_id)
    
def cmd_report(run_id: int, compare_run_id: int | None):
    init_db()
    print_report(run_id, compare_run_id=compare_run_id)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-as-judge eval harness")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run the eval pipeline")
    run_parser.add_argument("--notes", type=str, default=None)

    report_parser = sub.add_parser("report", help="Print report for a run")
    report_parser.add_argument("--run-id", type=int, required=True)
    report_parser.add_argument("--compare", type=int, default=None)
    
    args = parser.parse_args()
    
    if args.command == "run":
        cmd_run(notes=args.notes)
    elif args.command == "report":
        cmd_report(run_id=args.run_id, compare_run_id=args.compare)
