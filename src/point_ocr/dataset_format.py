"""Dataset record schemas and LLaMA-Factory / ShareGPT exporters."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from point_ocr.prompts import get_prompt


TaskName = Literal["POINT", "PAGE"]


@dataclass
class PointSample:
    """One training pair: marked image + task prompt + target text."""

    sample_id: str
    image_path: str
    task: TaskName
    target: str
    meta: dict[str, Any] = field(default_factory=dict)

    def user_text(self) -> str:
        return get_prompt(self.task)

    def to_sharegpt(self) -> dict[str, Any]:
        """LLaMA-Factory multimodal ShareGPT-style record."""
        return {
            "messages": [
                {
                    "role": "user",
                    "content": f"<image>{self.user_text()}",
                },
                {
                    "role": "assistant",
                    "content": self.target,
                },
            ],
            "images": [self.image_path],
            "metadata": {
                "sample_id": self.sample_id,
                "task": self.task,
                **self.meta,
            },
        }

    def to_openai_vl(self) -> dict[str, Any]:
        """Alternative OpenAI-style multimodal chat record."""
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": self.image_path},
                        {"type": "text", "text": self.user_text()},
                    ],
                },
                {"role": "assistant", "content": self.target},
            ],
            "metadata": {
                "sample_id": self.sample_id,
                "task": self.task,
                **self.meta,
            },
        }


def write_jsonl(path: Path, samples: Iterable[PointSample], *, fmt: str = "sharegpt") -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            rec = s.to_sharegpt() if fmt == "sharegpt" else s.to_openai_vl()
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def dump_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sample_to_dict(s: PointSample) -> dict[str, Any]:
    return asdict(s)
