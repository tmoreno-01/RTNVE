import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"benchmarks"))
from global_softwild_prompt_travel import make_plan

class GlobalSoftwildTests(unittest.TestCase):
    def test_schedule(self):
        p=make_plan()
        self.assertEqual(p,make_plan())
        self.assertEqual(p["loop"],"N")
        self.assertFalse(p["regional_masks"])
        self.assertEqual(len(p["base_prompts"]),4)
        self.assertEqual(len(p["prompts"]),24)
        self.assertEqual(len(p["frame_schedule"]),150)
        self.assertEqual(sum(s["dwell_frames"]+s["transition_frames"] for s in p["segments"]),150)
        prev=None
        for i,row in enumerate(p["frame_schedule"]):
            self.assertEqual(row["frame"],i)
            w={r["prompt_id"]:r["weight"] for r in row["weights"]}
            self.assertAlmostEqual(sum(w.values()),1.)
            self.assertTrue(all(0<v<=1 for v in w.values()))
            self.assertLessEqual(len(w),4)
            if prev is not None:
                delta=sum(abs(w.get(k,0)-prev.get(k,0)) for k in w.keys()|prev.keys())
                self.assertLess(delta,.35)
            prev=w
        self.assertEqual(p["frame_schedule"][-1]["base_from"],3)
        self.assertEqual(p["frame_schedule"][-1]["modifier_alpha"],1.)
        self.assertTrue(any(len(r["weights"])==4 for r in p["frame_schedule"]))
