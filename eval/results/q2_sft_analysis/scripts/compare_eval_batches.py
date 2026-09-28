"""Compare Q2 metrics between two prediction JSONLs (e.g. batch-1 vs batched eval).

  uv run python eval/results/q2_sft_analysis/scripts/compare_eval_batches.py \
    /tmp/q2_bs1/metrics/test_predictions.jsonl /tmp/q2_bs8/metrics/test_predictions.jsonl
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from point_ocr.q2_metrics import Q2Example, evaluate_q2, parse_ocr_mt_prediction  # noqa: E402


def load(path: str) -> dict:
    return {r["sample_id"]: r for r in (json.loads(l) for l in open(path, encoding="utf-8"))}


def main() -> None:
    a, b = load(sys.argv[1]), load(sys.argv[2])
    ids = sorted(set(a) & set(b))
    ident = sum(1 for i in ids if a[i]["prediction"] == b[i]["prediction"])
    src_ident = sum(
        1
        for i in ids
        if parse_ocr_mt_prediction(a[i]["prediction"]).source
        == parse_ocr_mt_prediction(b[i]["prediction"]).source
    )
    empty_agr = sum(1 for i in ids if (not a[i]["prediction"].strip()) == (not b[i]["prediction"].strip()))
    print(f"n={len(ids)}  identical={ident}  source_identical={src_ident}  empty_agreement={empty_agr}")

    def report(d):
        ex = [
            Q2Example(i, d[i]["prediction"], d[i]["target"], bool(d[i]["is_negative"]), d[i].get("prediction_raw"))
            for i in ids
        ]
        return evaluate_q2(ex)

    ra, rb = report(a), report(b)
    keys = [
        "source_hit_rate",
        "source_edit_similarity",
        "translation_chrf",
        "both_hit_rate",
        "xml_well_formed_rate",
        "strict_empty_rate",
        "effective_empty_rate",
        "hallucination_rate",
        "plain_text_rate",
        "format_leak_rate",
    ]
    print(f"{'metric':26s} {'batch1':>9s} {'batched':>9s} {'diff':>8s}")
    for k in keys:
        va, vb = getattr(ra, k), getattr(rb, k)
        print(f"{k:26s} {va:9.3f} {vb:9.3f} {vb - va:+8.3f}")


if __name__ == "__main__":
    main()
