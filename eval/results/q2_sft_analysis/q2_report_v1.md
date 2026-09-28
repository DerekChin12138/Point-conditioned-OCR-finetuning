# Q2 (OCR + EN→ZH) evaluation

- scored rows: **2764** (positive 2139 / negative 625)
- thresholds: source ≥ 0.85 edit-sim, translation ≥ 0.6 chrF

## Positives — localization
| metric | value |
|---|---|
| XML pair rate (has `<source>`) | 99.9% |
| XML well-formed rate | 98.4% |
| source hit rate | 89.3% |
| source edit similarity (mean) | 0.931 |
| positive over-extraction | 0.0% |

## Positives — translation
| metric | value |
|---|---|
| chrF2 (mean) | 0.341 |
| chrF++ (mean) | 0.285 |
| translation hit rate | 11.4% |
| both (source ∧ translation) hit | 10.8% |
| empty translation | 2.6% |
| CJK char ratio (mean) | 0.771 |
| length ratio pred/GT (mean) | 0.908 |
| unclosed `</translation>` | 1.5% |

## Negatives — empty discipline
| metric | value |
|---|---|
| strict empty (exactly `""`) | 54.1% |
| effective empty (+shell/bare `<source>`) | 65.8% |
| broken empty (shell/truncated, not clean) | 11.7% |
| hallucination (non-empty source) | 34.2% |
| plain text (no XML) | 9.1% |
| negative over-extraction | 0.0% |

format leak: 0.0%

## By bucket

| bucket | n | src hit | chrF2 | both hit | eff-empty | halluc |
|---|---|---|---|---|---|---|
| core_inner | 1500 | 94.8% | 0.349 | 10.9% | 0.0% | 0.0% |
| empty_clear | 375 | 0.0% | 0.000 | 0.0% | 64.3% | 35.7% |
| empty_special | 250 | 0.0% | 0.000 | 0.0% | 68.0% | 32.0% |
| multi_frag | 375 | 75.2% | 0.308 | 7.5% | 0.0% | 0.0% |
| real_labeled | 76 | 82.9% | 0.562 | 47.4% | 0.0% | 0.0% |
| semantic_group | 188 | 76.6% | 0.249 | 2.1% | 0.0% | 0.0% |
