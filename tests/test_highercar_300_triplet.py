import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"benchmarks"))
from highercar_300_triplet import REFERENCES, BASE, LOOP, WIDTH, HEIGHT, FPS, WINDOW, STRIDE, render_size, output_size
from animatediff_long_context import uniform_contexts

class HighercarTripletTests(unittest.TestCase):
    def test_presets_and_schedule(self):
        self.assertEqual((WIDTH,HEIGHT,FPS,WINDOW,STRIDE,LOOP),(1280,720,10,16,1,"R-P"))
        self.assertEqual(REFERENCES[0]["prompt"],BASE["prompt"])
        self.assertEqual([(r["frames"],r["seed"],r["steps"],r["guidance"],r["overlap"]) for r in REFERENCES],
                         [(300,144,20,5.,4),(300,144,20,5.,4),(300,144,20,4.5,8),(50,144,20,5.,4)])
        self.assertEqual(len({r["filename"] for r in REFERENCES}),4)
        self.assertEqual(REFERENCES[3]["prompt"],BASE["prompt"])
        self.assertEqual([render_size(r) for r in REFERENCES],[(1280,720)]*3+[(1600,904)])
        self.assertEqual([output_size(r) for r in REFERENCES],[(1280,720)]*3+[(1600,900)])
        for r in REFERENCES:
            self.assertEqual(r["negative"],BASE["negative"])
            for step in range(r["steps"]):
                windows=uniform_contexts(step,r["frames"],WINDOW,STRIDE,r["overlap"],LOOP)
                self.assertEqual({f for w in windows for f in w},set(range(r["frames"])))
                if r["frames"]==300:
                    self.assertEqual(len(windows),25 if r["overlap"]==4 else 37)
