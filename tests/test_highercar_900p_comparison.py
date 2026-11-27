import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"benchmarks"))
from highercar_900p_comparison import SUITE, refinement_schedule, calls_for, BASE_SIZE, NATIVE_SIZE, OUTPUT_SIZE
class ComparisonSuiteTests(unittest.TestCase):
    def test_presets(self):
        self.assertEqual([r["seed"] for r in SUITE],[1223,731,1889,1223,1223,731])
        self.assertEqual([r["mode"] for r in SUITE],["native"]*3+["multires"]*3)
        self.assertEqual(len({r["filename"] for r in SUITE}),6)
        self.assertEqual([r["refine_strength"] for r in SUITE[3:]],[.35,.5,.35])
        for ref in SUITE:
            self.assertEqual((ref["frames"],ref["guidance"],ref["loop"],ref["overlap"]),(70,5.4,"R-P",4))
            self.assertEqual(ref["prompt"],SUITE[0]["prompt"])
            self.assertEqual(ref["negative"],SUITE[0]["negative"])
            self.assertEqual((ref["native_width"],ref["native_height"]),NATIVE_SIZE)
        self.assertEqual(BASE_SIZE,(1280,720))
        self.assertEqual(OUTPUT_SIZE,(1600,900))
    def test_schedule(self):
        self.assertEqual(refinement_schedule(.35,10),(29,19))
        self.assertEqual(refinement_schedule(.5,12),(24,12))
        self.assertEqual([calls_for(n) for n in (20,10,12)],[120,60,72])
