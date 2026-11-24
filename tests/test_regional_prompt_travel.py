import math
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"benchmarks"))
from regional_prompt_travel import build_plan, frame_params, prompt_at, CircularContext, REGIONAL_LOOP_MODE
from animatediff_long_context import uniform_contexts

class RegionalPlanTests(unittest.TestCase):
    def test_periodic_and_seeded(self):
        p=build_plan()
        self.assertEqual(p,build_plan())
        self.assertNotEqual(p,build_plan(seed=111))
        self.assertEqual(frame_params(p,0),frame_params(p,60))
        self.assertEqual(len(p["schedule"]),60)
        for r in p["regions"]:
            for knot in r["knots"]:
                left,right,alpha=prompt_at(knot,r["knots"])
                self.assertEqual(alpha,0)
                self.assertEqual(right,(left+1)%3)
    def test_motion_strength_and_prompt_travel(self):
        p=build_plan()
        for i in range(3):
            rows=[x["regions"][i] for x in p["schedule"]]
            self.assertGreater(max(r["cx"] for r in rows)-min(r["cx"] for r in rows),.4)
            self.assertGreater(max(r["rx"] for r in rows)-min(r["rx"] for r in rows),.15)
            lo,hi=p["regions"][i]["strength_range"]
            self.assertTrue(all(lo<=r["strength"]<=hi and 0<=r["alpha"]<=1 for r in rows))
            self.assertEqual({r["left"] for r in rows},{0,1,2})
            self.assertTrue(all(r["rx"]>0 and r["ry"]>0 for r in rows))
    def test_circular_windows(self):
        self.assertEqual(REGIONAL_LOOP_MODE,"R-P")
        self.assertEqual(build_plan()["loop"],"R-P")
        class Stub:
            def forward(self): pass
        controller=CircularContext(Stub(),None,60)
        self.assertEqual(controller.loop,"R-P")
        for step in range(20):
            windows=uniform_contexts(step,60,16,1,8,controller.loop)
            self.assertEqual(len(windows),7)
            self.assertNotEqual(windows,uniform_contexts(step,60,16,1,8,"A"))
            self.assertEqual({f for w in windows for f in w},set(range(60)))
            self.assertTrue(any(any(b<a for a,b in zip(w,w[1:])) for w in windows))

    def test_refined_150(self):
        p=build_plan(150,9191,refined=True)
        self.assertEqual(p["loop"],"R-P")
        self.assertEqual(p["global_weight"],.20)
        self.assertEqual(build_plan()["global_weight"],.38)
        self.assertEqual(len(p["schedule"]),150)
        self.assertEqual(frame_params(p,0),frame_params(p,150))
        for i,r in enumerate(p["regions"]):
            self.assertEqual(len(r["prompts"]),4)
            rows=[x["regions"][i] for x in p["schedule"]]
            self.assertEqual({x["left"] for x in rows},set(range(4)))
            self.assertTrue(all(0<=x["alpha"]<=1 and x["rx"]>0 and x["ry"]>0 for x in rows))
        for step in range(20):
            windows=uniform_contexts(step,150,16,1,8,"R-P")
            self.assertEqual(len(windows),18)
            self.assertEqual({f for w in windows for f in w},set(range(150)))
            self.assertTrue(any(b<a for w in windows for a,b in zip(w,w[1:])))

    def test_masks_and_attention_cpu_if_available(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed in syntax-only CI")
        from regional_prompt_travel import make_masks, RegionalState, SpatialProcessor
        p=build_plan()
        m=make_masks(p,9,16)
        self.assertEqual(tuple(m.shape),(60,4,9,16))
        self.assertTrue(bool(torch.isfinite(m).all()))
        self.assertTrue(bool((m>=0).all()))
        torch.testing.assert_close(m.sum(dim=1),torch.ones_like(m[:,0]))
        self.assertGreater(float(m[:,0].min()),0)
        self.assertGreater(float((m[0]-m[20]).abs().mean()),.01)
        n=16
        bank=[torch.full((60,2,3),float(i+3)) for i in range(3)]
        state=RegionalState(m,bank)
        state.select(list(range(54,60))+list(range(10)))
        def original(attn,h,encoder_hidden_states=None,**kw):
            return h*0+encoder_hidden_states[:,0,:].unsqueeze(1)
        proc=SpatialProcessor(original,state)
        h=torch.zeros((2*n,144,3))
        emb=torch.cat([torch.ones((n,2,3)),torch.full((n,2,3),2.)])
        out=proc(None,h,encoder_hidden_states=emb)
        torch.testing.assert_close(out[:n],torch.ones_like(out[:n]))
        w=state.weights(144,out.device,out.dtype)
        expected=(w*torch.tensor([2.,3.,4.,5.])).sum(-1,keepdim=True).expand(-1,-1,3)
        torch.testing.assert_close(out[n:],expected)
        self.assertEqual(state.calls,1)

if __name__=="__main__":
    unittest.main()
