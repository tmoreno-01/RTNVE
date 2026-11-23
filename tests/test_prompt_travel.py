"""CPU-only regression tests for the random semantic prompt controller."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from prompt_travel import build_prompt_plan, transition_weight, invent_prompt
import random


class PromptTravelTests(unittest.TestCase):
    def test_fixed_seed_reproducibility(self):
        args = dict(frames=240, seed=9191, first_prompt="A city",
                    min_interval=50, max_interval=200, transition_frames=16)
        self.assertEqual(build_prompt_plan(**args), build_prompt_plan(**args))
        other = build_prompt_plan(**{**args, "seed": 9192})
        self.assertNotEqual(build_prompt_plan(**args), other)

    def test_long_default_uses_50_100_200_frame_intervals(self):
        plan = build_prompt_plan(frames=1000, seed=86, first_prompt="A city")
        for segment in plan["segments"][:-1]:
            self.assertIn(segment["end_exclusive"] - segment["start"], (50, 100, 200))
        self.assertEqual(len(plan["frame_schedule"]), 1000)

    def test_48_frame_smoke_has_multiple_semantic_states(self):
        plan = build_prompt_plan(frames=48, seed=9191, first_prompt="NYC street",
                                 min_interval=12, max_interval=16, transition_frames=6)
        self.assertGreaterEqual(len(plan["prompts"]), 3)
        self.assertEqual(set(record["frame"] for record in plan["frame_schedule"]), set(range(48)))
        for record in plan["frame_schedule"]:
            self.assertTrue(0 <= record["alpha"] <= 1)
            self.assertLess(record["from_prompt"], len(plan["prompts"]))
            self.assertLess(record["to_prompt"], len(plan["prompts"]))

    def test_midpoint_easing_and_endpoints(self):
        self.assertGreater(transition_weight(0, 8), 0)
        self.assertLess(transition_weight(0, 8), 1)
        self.assertEqual(transition_weight(7, 8), 1.0)
        self.assertEqual(transition_weight(3, 7), 0.5)

    def test_prompt_invention_does_not_repeat_immediately(self):
        rng = random.Random(44)
        prev = ""
        for _ in range(200):
            next_prompt = invent_prompt(rng, prev, "surreal documentary")
            self.assertNotEqual(prev, next_prompt)
            self.assertIn("surreal documentary", next_prompt)
            prev = next_prompt

    def test_zero_transitions_are_supported(self):
        plan = build_prompt_plan(frames=120, seed=2, first_prompt="A city",
                                 min_interval=50, max_interval=50, transition_frames=0)
        self.assertTrue(all(f["alpha"] == 0 for f in plan["frame_schedule"]))

    def test_bad_settings_fail_early(self):
        for kwargs in (
            {"frames": 0},
            {"transition_frames": 50},
            {"min_interval": 1},
            {"min_interval": 99, "max_interval": 50},
            {"first_prompt": ""},
        ):
            args = dict(frames=120, seed=2, first_prompt="A city")
            args.update(kwargs)
            with self.assertRaises(ValueError):
                build_prompt_plan(**args)

if __name__ == "__main__":
    unittest.main()
