"""python -m point_ocr ..."""

from point_ocr.cli import build_synth, marker_demo, run_eval


def main() -> None:
    import sys

    cmds = {
        "marker-demo": marker_demo,
        "build-synth": build_synth,
        "eval": run_eval,
    }
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print("Usage: python -m point_ocr [marker-demo|build-synth|eval] ...")
        raise SystemExit(2)
    cmd = sys.argv.pop(1)
    cmds[cmd]()


if __name__ == "__main__":
    main()
