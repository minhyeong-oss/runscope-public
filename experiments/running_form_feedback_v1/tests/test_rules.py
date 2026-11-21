from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "src/analyze_form.py"
SPEC = importlib.util.spec_from_file_location("analyze_form", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RuleTests(unittest.TestCase):
    def test_joint_angle_straight_leg(self) -> None:
        angle = MODULE.joint_angle(
            np.array([0.0, 0.0]), np.array([0.0, 1.0]), np.array([0.0, 2.0])
        )
        self.assertAlmostEqual(angle, 180.0)

    def test_braking_low_flexion_is_improve(self) -> None:
        rows = pd.DataFrame(
            {"knee_flexion_deg": [34, 35, 36, 37, 38], "joint_confidence": [0.95] * 5}
        )
        result = MODULE.classify_braking(rows)
        self.assertEqual(result.status, "IMPROVE")
        self.assertEqual(result.problem_code, "limited_stance_knee_flexion")

    def test_landing_inconsistent_tibia_is_improve(self) -> None:
        rows = pd.DataFrame(
            {
                "tibia_tilt_deg": [2, 3, 4, 9, 10],
                "joint_confidence": [0.95] * 5,
            }
        )
        result = MODULE.classify_landing(rows)
        self.assertEqual(result.status, "IMPROVE")
        self.assertEqual(result.problem_code, "landing_tibia_alignment")

    def test_braking_adequate_flexion_is_good(self) -> None:
        rows = pd.DataFrame(
            {"knee_flexion_deg": [41, 43, 45, 47, 49], "joint_confidence": [0.95] * 5}
        )
        self.assertEqual(MODULE.classify_braking(rows).status, "GOOD")

    def test_too_few_strides_is_unknown(self) -> None:
        rows = pd.DataFrame(
            {"knee_flexion_deg": [45, 45, 45, 45], "joint_confidence": [0.95] * 4}
        )
        self.assertEqual(MODULE.classify_braking(rows).status, "UNKNOWN")

    def test_remediation_is_only_attached_to_improve(self) -> None:
        improve = MODULE.classify_braking(
            pd.DataFrame(
                {"knee_flexion_deg": [35] * 5, "joint_confidence": [0.95] * 5}
            )
        )
        good = MODULE.classify_braking(
            pd.DataFrame(
                {"knee_flexion_deg": [45] * 5, "joint_confidence": [0.95] * 5}
            )
        )
        config = {
            "remediation": {
                "limited_stance_knee_flexion": [{"url": "https://example.test/video"}]
            }
        }
        attached = MODULE.attach_remediation([improve, good], config)
        self.assertEqual(len(attached[0]["remediation_links"]), 1)
        self.assertEqual(attached[1]["remediation_links"], [])


if __name__ == "__main__":
    unittest.main()
