from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BASE_SOURCE = (
    ROOT
    / "experiments/landing_kth_event_gate_treadmill_test_v1/src/render_test_summary.py"
)
EXPERIMENT = ROOT / "experiments/landing_kth_event_gate_treadmill_test_v2"


def main() -> None:
    spec = importlib.util.spec_from_file_location("treadmill_test_v2_render", BASE_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {BASE_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EXPERIMENT = EXPERIMENT
    module.REPORT = EXPERIMENT / "reports/TEST_REPORT.json"
    module.OUTPUTS = EXPERIMENT / "reports/frozen_test_outputs.npz"
    module.main()


if __name__ == "__main__":
    main()
