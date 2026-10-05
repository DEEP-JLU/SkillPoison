from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import load_config
from core.pipeline import SkillPoisonPipeline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--replay", action="store_true")
    arguments = parser.parse_args()
    try:
        result = SkillPoisonPipeline(load_config(arguments.config)).run(input_path=arguments.input, output_directory_override=arguments.out, replay=arguments.replay)
    except (ValueError, RuntimeError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(f"candidates={result['n_candidates']} selected={result['n_selected']} records={result['n_records']}")
    print(f"J*={result['J_star']} b_j={result['allocation']['b_j']}")
    print(f"output={result['output_directory']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
