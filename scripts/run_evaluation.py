from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.evaluation.audit import audit_record, merge_control_records
from core.evaluation.metrics import summarize
from core.utils.io import read_jsonl, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", required=True)
    parser.add_argument("--control", default=None)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    try:
        attack_records = read_jsonl(arguments.attack)
        control_records = read_jsonl(arguments.control) if arguments.control else None
        audits = [audit_record(row) for row in
                  merge_control_records(attack_records, control_records)]
        summary = summarize(audits)
        write_json(arguments.out, summary)
    except (ValueError, KeyError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output={arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
