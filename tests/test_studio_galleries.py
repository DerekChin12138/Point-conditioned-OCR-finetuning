from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval" / "marker_studio"))
sys.path.insert(0, str(ROOT / "src"))

from build_galleries import select_q1_test_pages  # noqa: E402
from point_ocr.coord_condition import coord_sharegpt_row, render_path_for_meta  # noqa: E402
from server import _lora_base_model, _resolve_sample  # noqa: E402


def test_select_q1_covers_templates_and_pools():
    rows = []
    for i, tmpl in enumerate(["01_article_twocol", "14_magazine_3col", "06_dense_table_cells"]):
        for pool in ["core_inner", "empty_clear"]:
            pid = f"{pool}__{tmpl}__{i}"
            rows.append(
                {
                    "metadata": {
                        "page_id": pid,
                        "pool_id": pool,
                        "template_stem": tmpl,
                        "chrome_family": "none" if pool == "core_inner" else "browser",
                    }
                }
            )
    picked = select_q1_test_pages(rows, max_pages=10)
    stems = {m["template_stem"] for _, m in picked}
    pools = {m["pool_id"] for _, m in picked}
    assert stems == {"01_article_twocol", "14_magazine_3col", "06_dense_table_cells"}
    assert "core_inner" in pools
    assert "empty_clear" in pools


def test_coord_sharegpt_uses_unmarked_prompt():
    row = coord_sharegpt_row(
        sample_id="t",
        image_path="/tmp/page.png",
        target="Hello",
        px=10,
        py=20,
        image_w=100,
        image_h=200,
        meta={"pool_id": "core_inner", "page_id": "core_inner__x__1"},
    )
    user = row["messages"][0]["content"]
    assert user.startswith("<image>")
    assert "magenta" not in user.lower()
    assert '"point_2d": [10, 20]' in user
    assert "pixel" in user.lower()
    assert row["images"] == ["/tmp/page.png"]
    assert row["metadata"]["prompt_key"] == "coord_q1"
    assert row["metadata"]["marker"] == "none"
    assert row["metadata"]["point_2d"] == [10, 20]
    assert "no mark" not in user.lower()


def test_render_path_for_meta():
    p = render_path_for_meta(
        {"pool_id": "core_inner", "page_id": "core_inner__a__1"},
        pools_root=Path("/data/pools_q"),
    )
    assert p == Path("/data/pools_q/core_inner/renders/core_inner__a__1.png")


def test_lora_base_prefers_adapter_config(tmp_path: Path):
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "models/Qwen3.5-0.8B"}),
        encoding="utf-8",
    )
    # Relative path is resolved against the repo if that dir exists.
    got = _lora_base_model(str(tmp_path))
    assert "Qwen" in got or "Ovis" in got or got.endswith("Qwen3.5-0.8B")


def test_resolve_sample_real_png():
    real = ROOT / "data/real_world_sample"
    pngs = list(real.glob("*.png"))
    if not pngs:
        return
    name = pngs[0].name
    assert _resolve_sample(name) == (real / name).resolve()
    assert _resolve_sample(f"real/{name}") == (real / name).resolve()


def test_list_bases_includes_merged_sft():
    from server import _list_bases  # noqa: E402

    bases = _list_bases()
    ids = {b["id"] for b in bases}
    assert "Qwen3.5-0.8B" in ids
    merged = ROOT / "checkpoints" / "q1_withreal_merged"
    if (merged / "config.json").is_file() and (merged / "model.safetensors").is_file():
        assert "q1_withreal_merged" in ids
        path = next(b["path"] for b in bases if b["id"] == "q1_withreal_merged")
        assert Path(path).resolve() == merged.resolve()


def test_signal_multi_frag_gallery_resolve():
    gallery = ROOT / "data/studio_galleries" / "signal_multi_frag"
    imgs = [
        p
        for p in gallery.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    ]
    if not imgs:
        return
    name = imgs[0].name
    got = _resolve_sample(f"signal_multi_frag/{name}")
    assert got == (gallery / name)
    assert got.is_file()
    assert got.samefile(gallery / name)
