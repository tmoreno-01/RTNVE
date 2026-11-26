import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"benchmarks"))
from reference_match_triplet import REFERENCES, emphasis_parts, LOOP, WIDTH, HEIGHT, FPS, WINDOW, STRIDE
from animatediff_long_context import uniform_contexts

class ReferenceTripletTests(unittest.TestCase):
    def test_presets(self):
        self.assertEqual((WIDTH,HEIGHT,FPS,WINDOW,STRIDE,LOOP),(1280,720,10,16,1,"R-P"))
        self.assertEqual([(r["seed"],r["frames"],r["steps"],r["guidance"],r["overlap"]) for r in REFERENCES],
                         [(1223,100,20,6.,6),(144,100,20,5.,4),(21,80,22,5.,4)])
        self.assertEqual([r["negative"] for r in REFERENCES],["nsfw"]+["nsfw, naked, duplicates, zoom, pan"]*2)
        for ref,expected in zip(REFERENCES,[[1.4],[1.2],[1.1,1.4]]):
            parts=emphasis_parts(ref["prompt"])
            self.assertEqual([w for _,w in parts if w!=1.],expected)
            self.assertNotIn("(", "".join(t for t,w in parts))
            self.assertNotIn("<lora:",ref["prompt"])
            for step in range(ref["steps"]):
                windows=uniform_contexts(step,ref["frames"],WINDOW,STRIDE,ref["overlap"],LOOP)
                self.assertEqual({i for window in windows for i in window},set(range(ref["frames"])))
                # R-P can have an exact boundary alignment on some steps.
            self.assertTrue(any(b<a for step in range(ref["steps"]) for w in uniform_contexts(step,ref["frames"],WINDOW,STRIDE,ref["overlap"],LOOP) for a,b in zip(w,w[1:])))
