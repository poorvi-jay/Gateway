"""
Measures how well the detector catches PII, and how fast it is.

  recall    = of all real PII in the dataset, how much did we catch?
              (a miss = PII that would have LEAKED to the LLM)
  precision = of everything we flagged, how much was really PII?
              (a false flag = harmless text masked unnecessarily)

For a privacy gateway recall matters more: a miss is a leak, a false flag only
costs some answer quality. A span counts as caught if any detection overlaps
it (any entity type), because for leak prevention what matters is that the
value was masked, not which label it got. Per-type numbers are also reported.

Run from the project root:   python -m eval.evaluate
"""
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("SPACY_MODEL", "en_core_web_lg")

from app.detector import Detector  # noqa: E402
from app.policy import load_policy  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    policy = load_policy(ROOT / "policy.yaml")
    detector = Detector(policy)
    rows = [json.loads(l) for l in (ROOT / "eval" / "dataset.jsonl").read_text().splitlines() if l.strip()]

    tp_gold = fn = tp_pred = fp = 0
    per_type = defaultdict(lambda: [0, 0])  # type -> [caught, total]
    misses, false_flags, latencies = [], [], []

    detector.detect("warm up the model")
    for row in rows:
        text = row["text"]
        gold = []
        for etype, value in row["pii"]:
            s = text.index(value)
            gold.append((etype, s, s + len(value), value))

        t = time.perf_counter()
        preds = detector.detect(text)
        latencies.append((time.perf_counter() - t) * 1000)

        for etype, s, e, value in gold:
            hit = any(p.start < e and s < p.end for p in preds)
            per_type[etype][1] += 1
            if hit:
                tp_gold += 1
                per_type[etype][0] += 1
            else:
                fn += 1
                misses.append(f"{etype:<14} {value!r:<28} in: {text}")
        for p in preds:
            if any(p.start < e and s < p.end for _, s, e, _ in gold):
                tp_pred += 1
            else:
                fp += 1
                false_flags.append(f"{p.entity_type:<14} {p.text!r:<28} in: {text}")

    recall = tp_gold / (tp_gold + fn) if tp_gold + fn else 1.0
    precision = tp_pred / (tp_pred + fp) if tp_pred + fp else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    lat = sorted(latencies)

    print(f"Model: {os.environ['SPACY_MODEL']}   Sentences: {len(rows)}   "
          f"PII spans: {tp_gold + fn}")
    print(f"\nRecall    {recall:6.1%}   (PII caught; misses would leak)")
    print(f"Precision {precision:6.1%}   (flags that were really PII)")
    print(f"F1        {f1:6.1%}")
    print(f"\nDetection latency per message: mean {statistics.mean(lat):.1f} ms, "
          f"p95 {lat[int(0.95 * (len(lat) - 1))]:.1f} ms")
    print("\nRecall by type:")
    for etype, (c, n) in sorted(per_type.items()):
        print(f"  {etype:<14} {c}/{n}")
    print("\nMissed (false negatives):")
    print("\n".join("  " + m for m in misses) or "  none")
    print("\nFalse flags (false positives):")
    print("\n".join("  " + f for f in false_flags) or "  none")


if __name__ == "__main__":
    main()
