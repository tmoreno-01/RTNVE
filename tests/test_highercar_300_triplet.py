import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"benchmarks"))
from highercar_300_triplet import REFERENCES, HIGHRES_REFERENCES, HIGHRES_70_REFERENCES, RETURN_720P_REFERENCES, BASE, LOOP, WIDTH, HEIGHT, FPS, WINDOW, STRIDE, render_size, output_size
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

    def test_highres_triplet(self):
        self.assertEqual([(r["frames"],r["seed"],r["steps"],r["guidance"],r["overlap"]) for r in HIGHRES_REFERENCES],
                         [(50,1223,20,5.,4),(50,731,20,5.,4),(50,2849,20,4.8,6)])
        self.assertEqual(HIGHRES_REFERENCES[0]["prompt"],BASE["prompt"])
        self.assertEqual(len({r["filename"] for r in REFERENCES+HIGHRES_REFERENCES}),7)
        for r in HIGHRES_REFERENCES:
            self.assertTrue(r["prompt"].startswith(BASE["prompt"]))
            self.assertEqual(r["negative"],BASE["negative"])
            self.assertEqual(render_size(r),(1600,904))
            self.assertEqual(output_size(r),(1600,900))
            for step in range(r["steps"]):
                windows=uniform_contexts(step,r["frames"],WINDOW,STRIDE,r["overlap"],LOOP)
                self.assertEqual({f for w in windows for f in w},set(range(50)))
                self.assertEqual(len(windows),4 if r["overlap"]==4 else 5)

    def test_highres_70_triplet(self):
        self.assertEqual(len(HIGHRES_70_REFERENCES),3)
        self.assertEqual(len({r["filename"] for r in REFERENCES+HIGHRES_REFERENCES+HIGHRES_70_REFERENCES}),10)
        for previous, current in zip(HIGHRES_REFERENCES,HIGHRES_70_REFERENCES):
            self.assertEqual(current,dict(previous,frames=70,filename=previous["filename"].replace("_50f_","_70f_")))
            self.assertEqual(render_size(current),(1600,904))
            self.assertEqual(output_size(current),(1600,900))
            for step in range(current["steps"]):
                windows=uniform_contexts(step,70,WINDOW,STRIDE,current["overlap"],LOOP)
                self.assertEqual({f for w in windows for f in w},set(range(70)))
                self.assertEqual(len(windows),6 if current["overlap"]==4 else 7)

    def test_return_720p_triplet(self):
        self.assertEqual([(r["frames"],r["seed"],r["steps"],r["guidance"],r["overlap"]) for r in RETURN_720P_REFERENCES],
                         [(300,1223,20,5.,4),(300,731,20,5.,4),(300,1889,20,5.,4)])
        self.assertEqual([r["prompt"] for r in RETURN_720P_REFERENCES],
                         [BASE["prompt"],BASE["prompt"],BASE["prompt"]+", deep shadows, moody contrast"])
        self.assertEqual(len({r["filename"] for r in REFERENCES+HIGHRES_REFERENCES+HIGHRES_70_REFERENCES+RETURN_720P_REFERENCES}),13)
        for r in RETURN_720P_REFERENCES:
            self.assertEqual(r["negative"],BASE["negative"])
            self.assertEqual(render_size(r),(1280,720))
            self.assertEqual(output_size(r),(1280,720))
            for step in range(r["steps"]):
                windows=uniform_contexts(step,r["frames"],WINDOW,STRIDE,r["overlap"],LOOP)
                self.assertEqual({f for w in windows for f in w},set(range(300)))
                self.assertEqual(len(windows),25)
