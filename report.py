from store import get_results


def print_report(run_id: int, compare_run_id: int | None = None):
    results = get_results(run_id)
    if not results:
        print(f"No results found for run #{run_id}")
        return

    # per-question table
    print(f"{'='*80}")
    print(f"Run #{run_id} — {len(results)} questions")
    print(f"{'='*80}")
    print(f"{'Q':<4} {'Status':<6} {'F':>5} {'AR':>5} {'CP':>5}  Reasoning")
    print(f"{'-'*80}")

    for i, r in enumerate(results, 1):
        status = "PASS" if r["passed"] else "FAIL"
        reasoning = r["reasoning"][:60] + "..." if len(r["reasoning"]) > 60 else r["reasoning"]
        print(f"{i:<4} {status:<6} {r['faithfulness']:>5.2f} {r['answer_relevance']:>5.2f} {r['context_precision']:>5.2f}  {reasoning}")

    # aggregate summary
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    mean_f  = sum(r["faithfulness"]      for r in results) / total
    mean_ar = sum(r["answer_relevance"]  for r in results) / total
    mean_cp = sum(r["context_precision"] for r in results) / total

    print(f"{'='*80}")
    print(f"Pass rate : {passed}/{total} ({100*passed/total:.1f}%)")
    print(f"Faithfulness      : {mean_f:.3f}")
    print(f"Answer Relevance  : {mean_ar:.3f}")
    print(f"Context Precision : {mean_cp:.3f}")
    print(f"{'='*80}")

    if compare_run_id is None:
        return

    # delta vs previous run
    prev_results = get_results(compare_run_id)
    if not prev_results:
        print(f"No results found for comparison run #{compare_run_id}")
        return

    prev_map = {r["question"]: r for r in prev_results}

    print(f"\nDelta — Run #{compare_run_id} → Run #{run_id}")
    print(f"{'-'*80}")
    print(f"{'Q':<4} {'Δ F':>6} {'Δ AR':>6} {'Δ CP':>6}  Status")
    print(f"{'-'*80}")

    regressions = 0
    improvements = 0

    for i, r in enumerate(results, 1):
        prev = prev_map.get(r["question"])
        if prev is None:
            continue

        df  = r["faithfulness"]      - prev["faithfulness"]
        dar = r["answer_relevance"]  - prev["answer_relevance"]
        dcp = r["context_precision"] - prev["context_precision"]
        mean_delta = (df + dar + dcp) / 3

        if mean_delta < -0.05:
            tag = "REGRESSED"
            regressions += 1
        elif mean_delta > 0.05:
            tag = "IMPROVED"
            improvements += 1
        else:
            tag = ""

        print(f"{i:<4} {df:>+6.2f} {dar:>+6.2f} {dcp:>+6.2f}  {tag}")

    print(f"{'-'*80}")
    print(f"Regressions: {regressions}  Improvements: {improvements}")
    print(f"{'='*80}")
