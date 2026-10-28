from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.evaluation.audit import audit_record, merge_control_records
from core.utils.io import read_jsonl, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--control", default=None)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    try:
        attack_records = read_jsonl(arguments.input)
        control_records = read_jsonl(arguments.control) if arguments.control else None
        audits = [audit_record(row) for row in
                  merge_control_records(attack_records, control_records)]
        write_jsonl(arguments.out, audits)
    except (ValueError, KeyError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(f"records={len(audits)} target_behavior={sum(1 for a in audits if a['target_behavior_present'])} adopted={sum(1 for a in audits if a['adopted'])} attributable={sum(1 for a in audits if a['attributable_failure'])}")
    print(f"output={arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
