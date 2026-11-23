"""RTNVE temporal context scheduler tests. No GPU/dependencies required."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from animatediff_long_context import ordered_halving, uniform_contexts, frame_condition_indices


class UniformContextTests(unittest.TestCase):
    def test_a1111_phase_sequence(self):
        self.assertEqual(ordered_halving(0), 0)
        self.assertEqual(ordered_halving(1), 0.5)
        self.assertEqual(ordered_halving(2), 0.25)
        self.assertEqual(ordered_halving(3), 0.75)

    def test_screen_settings_cover_all_100_frames_at_every_step(self):
        for step in range(20):
            w = uniform_contexts(step, 100, 13, 1, 6, "R+P")
            self.assertEqual(len(w), 14)
            self.assertTrue(all(len(ids) == 13 for ids in w))
            hits = [i for ids in w for i in ids]
            self.assertEqual(set(hits), set(range(100)))
            self.assertEqual(len(hits), 14 * 13)

    def test_known_first_step_windows_match_original_generator(self):
        w = uniform_contexts(0, 100, 13, 1, 6, "R+P")
        self.assertEqual(w[0], list(range(13)))
        self.assertEqual(w[-1], list(range(91, 100)) + list(range(4)))

    def test_step_shift_changes_contexts(self):
        a = uniform_contexts(0, 100, 13, 1, 6, "R+P")
        b = uniform_contexts(1, 100, 13, 1, 6, "R+P")
        self.assertNotEqual(a[0], b[0])
        self.assertEqual(b[0], list(range(50, 63)))

    def test_one_window(self):
        self.assertEqual(
            uniform_contexts(0, 13, 13, 1, 6, "R+P"), [list(range(13))]
        )

    def test_invalid_overlap_rejected(self):
        with self.assertRaises(ValueError):
            uniform_contexts(0, 100, 13, 1, 13, "R+P")

    def test_N_never_wraps_or_connects_final_to_first_frame(self):
        # The non-looping mode must not ever place frame 0 immediately after
        # the final frame in a temporal attention window.
        for n, window, overlap in [(48, 16, 8), (100, 16, 8),
                                   (48, 13, 6), (100, 13, 6)]:
            for step in range(64):
                windows = uniform_contexts(step, n, window, 1, overlap, "N")
                self.assertTrue(windows)
                self.assertEqual(set(f for w in windows for f in w), set(range(n)))
                self.assertTrue(all(len(w) == window for w in windows))
                for w in windows:
                    self.assertEqual(w, list(range(w[0], w[0] + window)))

    def test_N_fills_upstream_boundary_gap(self):
        # The upstream A1111 algorithm can skip early frames at steps 4,8,
        # etc., leaving uninitialised predictions. RTNVE repairs coverage.
        for step in [4, 8, 16, 32]:
            windows = uniform_contexts(step, 48, 16, 1, 8, "N")
            covered = {f for w in windows for f in w}
            self.assertEqual(covered, set(range(48)))
            self.assertIn(0, covered)

    def test_overnight_1000_frame_nonloop_context_coverage(self):
        for step in range(20):
            windows = uniform_contexts(step, 1000, 16, 1, 8, "N")
            self.assertGreater(len(windows), 100)
            self.assertEqual({f for w in windows for f in w}, set(range(1000)))
            self.assertTrue(all(w == list(range(w[0], w[0]+16)) for w in windows))

    def test_N_requires_stride_one(self):
        with self.assertRaises(ValueError):
            uniform_contexts(0, 100, 16, 2, 8, "N")

    def test_different_lengths_and_overlaps(self):
        for n,b,o in [(48,13,6),(100,16,8),(32,12,4)]:
            for step in range(8):
                all_frames = {i for w in uniform_contexts(step,n,b,1,o,"R+P") for i in w}
                self.assertEqual(all_frames,set(range(n)))

    def test_closed_loop_modes_still_available_for_regression(self):
        for loop in ("R-P", "R+P", "A"):
            windows = uniform_contexts(0, 48, 16, 1, 8, loop)
            self.assertEqual(set(f for w in windows for f in w), set(range(48)))

    def test_diffusers_cfg_text_embeddings_follow_each_window(self):
        # Exactly the first failure case: 48 frames with context 13,
        # classifier-free guidance produces TWO sets of prompt embeddings.
        frames = 48
        context = list(range(13))
        selection = frame_condition_indices(2, frames, context)
        self.assertEqual(len(selection), 26)
        self.assertEqual(selection[:13], context)
        self.assertEqual(selection[13:], list(range(48, 61)))
        # These indices align with the two [B,C,13,H,W] UNet inputs.

    def test_shifted_temporal_window_selects_matching_prompt_frames(self):
        context = [43, 44, 45, 46, 47, 0, 1, 2, 3, 4, 5, 6, 7]
        selection = frame_condition_indices(2, 48, context)
        self.assertEqual(selection[:13], context)
        self.assertEqual(selection[13:], [x + 48 for x in context])

    def test_context_alignment_rejects_bad_frame(self):
        with self.assertRaises(ValueError):
            frame_condition_indices(2, 48, [48])
        with self.assertRaises(ValueError):
            frame_condition_indices(0, 48, [1])


if __name__ == "__main__":
    unittest.main()
