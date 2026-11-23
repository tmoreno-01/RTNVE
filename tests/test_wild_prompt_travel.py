"""CPU-only checks for the overnight surreal prompt schedule."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"benchmarks"))
from wild_prompt_travel import make_plan, make_prompt
import random

class OvernightWildTravelTests(unittest.TestCase):
    def test_one_thousand_frames_every_frame_is_conditioned(self):
        plan=make_plan(frames=1000,seed=9191,min_interval=60,max_interval=140,transition_frames=32)
        self.assertEqual(len(plan["frame_schedule"]),1000)
        self.assertGreaterEqual(len(plan["prompts"]),7)
        self.assertEqual([x["frame"] for x in plan["frame_schedule"]],list(range(1000)))
        self.assertEqual(plan["segments"][0]["start"],0)
        self.assertEqual(plan["segments"][-1]["end_exclusive"],1000)
        for a,b in zip(plan["segments"],plan["segments"][1:]):
            self.assertEqual(a["end_exclusive"],b["start"])
        for segment in plan["segments"][:-1]:
            self.assertGreaterEqual(segment["end_exclusive"]-segment["start"],60)
            self.assertLessEqual(segment["end_exclusive"]-segment["start"],140)
            self.assertEqual(segment["end_exclusive"]-segment["transition_start"],32)
        for i,entry in enumerate(plan["frame_schedule"]):
            self.assertTrue(0.0<=entry["alpha"]<=1.0)
            self.assertLess(entry["from_prompt"],len(plan["prompts"]))
            self.assertLess(entry["to_prompt"],len(plan["prompts"]))

    def test_smooth_32_frame_changes_not_hard_cuts(self):
        plan=make_plan(1000,9191,60,140,32)
        for seg in plan["segments"][:-1]:
            values=[plan["frame_schedule"][i]["alpha"] for i in range(
                seg["transition_start"],seg["end_exclusive"])]
            self.assertEqual(len(values),32)
            self.assertEqual(values[-1],1)
            self.assertTrue(all(values[j]<values[j+1] for j in range(31)))
        self.assertTrue(all(x["alpha"]==0 for x in plan["frame_schedule"][
            plan["segments"][-1]["start"]:]))

    def test_no_lora_and_genuinely_different_subjects(self):
        plan=make_plan(1000,9191,60,140,32)
        self.assertEqual(len(set(plan["prompts"])),len(plan["prompts"]))
        self.assertTrue(all("<lora:" not in s.lower() for s in plan["prompts"]))
        self.assertTrue(all(len(s)>100 for s in plan["prompts"]))
        self.assertTrue(any("New York" not in s for s in plan["prompts"]))

    def test_exact_seed_repeatable(self):
        p=make_plan(1000,9191)
        self.assertEqual(p,make_plan(1000,9191))
        self.assertNotEqual(p,make_plan(1000,9192))

    def test_invalid_parameters(self):
        for params in [(0,9191,60,140,32),(1000,9191,10,20,32),
                       (1000,9191,100,60,32),(1000,9191,60,140,0)]:
            with self.assertRaises(ValueError):
                make_plan(*params)

    def test_context_script_can_be_imported_without_torch_cuda(self):
        # Legacy 200f cap does not affect the dedicated overnight renderer.
        import overnight_1000
        self.assertTrue(hasattr(overnight_1000,"decode_to_mp4"))
        self.assertTrue(hasattr(overnight_1000,"gpu_preflight"))

if __name__=="__main__":
    unittest.main()
