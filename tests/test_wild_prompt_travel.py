"""CPU-only tests for the 1000-frame surreal prompt schedule."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from wild_prompt_travel import make_plan


class OvernightWildTravelTests(unittest.TestCase):
    def test_thousand_frames_contiguous_and_semantically_varied(self):
        plan = make_plan(frames=1000, seed=9191,
                         min_interval=60, max_interval=140,
                         transition_min=20, transition_max=60)
        self.assertEqual(len(plan["frame_schedule"]), 1000)
        self.assertGreaterEqual(len(plan["prompts"]), 7)
        self.assertEqual([x["frame"] for x in plan["frame_schedule"]], list(range(1000)))
        self.assertEqual(plan["segments"][0]["start"], 0)
        self.assertEqual(plan["segments"][-1]["end_exclusive"], 1000)
        self.assertEqual(len(set(plan["prompts"])), len(plan["prompts"]))
        self.assertTrue(all("<lora:" not in p.lower() for p in plan["prompts"]))

        changes = []
        for a, b in zip(plan["segments"], plan["segments"][1:]):
            self.assertEqual(a["end_exclusive"], b["start"])
        for segment in plan["segments"][:-1]:
            duration = segment["end_exclusive"] - segment["start"]
            transition = segment["transition_frames"]
            self.assertGreaterEqual(duration, 60)
            self.assertLessEqual(duration, 140)
            self.assertTrue(20 <= transition <= 60)
            self.assertLess(transition, duration)
            self.assertEqual(segment["end_exclusive"] - segment["transition_start"], transition)
            changes.append(transition)

        # Verify that it is not just one transition length repeated everywhere.
        self.assertGreaterEqual(len(set(changes)), 3)
        self.assertGreaterEqual(len(set(p.split(",")[1] for p in plan["prompts"])), 3)

    def test_each_transition_is_smooth_and_different(self):
        plan = make_plan(1000, 9191, 60, 140, 20, 60)
        for seg in plan["segments"][:-1]:
            n = seg["transition_frames"]
            values = [plan["frame_schedule"][i]["alpha"] for i in range(
                seg["transition_start"], seg["end_exclusive"])]
            self.assertEqual(len(values), n)
            self.assertEqual(values[-1], 1.0)
            self.assertTrue(all(values[i] < values[i+1] for i in range(n-1)))
        self.assertTrue(all(x["alpha"] == 0 for x in plan["frame_schedule"][
            plan["segments"][-1]["start"]:]))

    def test_seed_is_reproducible(self):
        a = make_plan(1000, 9191)
        self.assertEqual(a, make_plan(1000, 9191))
        self.assertNotEqual(a, make_plan(1000, 9192))

    def test_transition_duration_never_exceeds_shortest_segment(self):
        plan = make_plan(200, 7, 60, 60, 20, 60)
        for seg in plan["segments"][:-1]:
            self.assertLess(seg["transition_frames"], 60)

    def test_invalid_parameters(self):
        bad = [
            (0, 9191, 60, 140, 20, 60),
            (1000, 9191, 10, 20, 20, 60),
            (1000, 9191, 100, 60, 20, 60),
            (1000, 9191, 60, 140, 0, 60),
            (1000, 9191, 60, 140, 61, 61),
            (1000, 9191, 60, 140, 40, 20),
        ]
        for params in bad:
            with self.subTest(params=params), self.assertRaises(ValueError):
                make_plan(*params)

    def test_overnight_renderer_is_importable_without_cuda(self):
        import overnight_1000
        self.assertTrue(hasattr(overnight_1000, "decode_to_mp4"))
        self.assertTrue(hasattr(overnight_1000, "gpu_preflight"))


if __name__ == "__main__":
    unittest.main()
