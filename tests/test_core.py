import importlib.util, sys, tempfile, os, unittest
from fractions import Fraction
import torch, av, numpy as np

spec = importlib.util.spec_from_file_location("core", os.path.join(os.path.dirname(__file__), "..", "core.py"))
core = importlib.util.module_from_spec(spec); sys.modules["core"] = core; spec.loader.exec_module(core)


class FakeVideo:
    def __init__(self, path, fps): self.path, self.fps = path, fps
    def get_stream_source(self): return self.path
    def get_active_trim_window(self): return 0.0, 0.0
    def get_frame_rate(self): return Fraction(self.fps)
    def get_frame_count(self):
        with av.open(self.path) as c: return c.streams.video[0].frames


def make_video(path, n=48, fps=24):
    with av.open(path, "w") as c:
        s = c.add_stream("mpeg4", rate=fps); s.width, s.height, s.pix_fmt = 64, 48, "yuv420p"
        for i in range(n):
            f = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i * 5 % 255, np.uint8), format="rgb24")
            for p in s.encode(f): c.mux(p)
        for p in s.encode(): c.mux(p)


class Tests(unittest.TestCase):
    def test_timestamp_independent_of_frame_rate(self):
        frames = torch.rand(81, 32, 32, 3)
        for fps in (16.0, 24.0, 60.0):
            f, idx, total, nat = core.sample_images(frames, 9, "uniform", fps)
            ts = core.resolve_timestamps(idx, total, nat, 5.0)
            self.assertEqual(ts, [i * 5.0 / 81 for i in idx])
        # without override the old behaviour (fps-dependent) is kept
        _, idx, total, nat = core.sample_images(frames, 9, "uniform", 16.0)
        self.assertAlmostEqual(core.resolve_timestamps(idx, total, nat, 0.0)[-1], 80 / 16.0)

    def test_interpolated_clip_keeps_original_time(self):
        orig = core.sample_images(torch.rand(40, 16, 16, 3), 4, "uniform", 8.0)   # 5s @8fps
        interp = core.sample_images(torch.rand(80, 16, 16, 3), 4, "uniform", 16.0)  # 5s @16fps
        a = core.resolve_timestamps(orig[1], orig[2], orig[3], 5.0)
        b = core.resolve_timestamps(interp[1], interp[2], interp[3], 5.0)
        for x, y in zip(a, b): self.assertAlmostEqual(x, y, delta=0.1)

    def test_grid_above_8(self):
        for g in (9, 16, 32):
            frames = torch.rand(2000, 48, 64, 3) if g == 32 else torch.rand(300, 48, 64, 3)
            f, idx, total, nat = core.sample_images(frames, g * g, "uniform", 24.0)
            sheet = core.render_contact_sheet(f, core.resolve_timestamps(idx, total, nat, 10.0),
                                              g, g, 0, 0, "fit", 0, True, "index_timestamp", 160)
            self.assertEqual(sheet.shape[0], 1)
            self.assertGreaterEqual(sheet.shape[2] // g, 160)
            self.assertAlmostEqual(sheet.shape[1] / sheet.shape[2], 48 / 64, delta=0.01)

    def test_too_small_gives_clear_error(self):
        f = torch.rand(100, 48, 64, 3)
        fr, idx, total, nat = core.sample_images(f, 256, "uniform", 24.0)
        with self.assertRaises(ValueError):
            core.render_contact_sheet(fr, nat, 16, 16, 128, 96, "fit", 0, True, "none", 0)

    def test_file_video_scaled_timestamps(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "v.mp4"); make_video(p, 48, 24)
            for mode in ("uniform", "motion_weighted"):
                fr, idx, total, nat = core.sample_video(FakeVideo(p, 24), 16, mode)
                self.assertEqual(fr.shape[0], 16)
                self.assertEqual(idx[0], 0)
                ts = core.resolve_timestamps(idx, total, nat, 2.0)
                self.assertAlmostEqual(ts[0], 0.0)
                self.assertLess(ts[-1], 2.0)
                sheet = core.render_contact_sheet(fr, ts, 4, 4, 0, 0, "crop", 2, False, "timestamp", 120)
                self.assertEqual(sheet.shape[-1], 3)

    def test_layouts(self):
        self.assertEqual(core.choose_layout(40), (8, 5))
        self.assertEqual(core.choose_layout(49), (7, 7))
        self.assertEqual(core.choose_layout(16), (4, 4))
        self.assertEqual(core.choose_layout(41), (7, 6))
        self.assertEqual(core.choose_layout(1), (1, 1))
        self.assertEqual(core.probe_source(None, torch.rand(48, 8, 8, 3), 24.0), (48, 2.0))

    def test_five_second_clip_at_8_per_second(self):
        frames = torch.rand(120, 48, 64, 3)  # 5 s @ 24 fps
        total, dur = core.probe_source(None, frames, 24.0)
        cols, rows, n = core.plan_layout(total, dur, 8.0)
        self.assertEqual((cols, rows, n), (8, 5, 40))
        f, idx, tot, nat = core.sample_images(frames, n, "uniform", 24.0, "interval")
        self.assertEqual(len(idx), 40)
        ts = core.resolve_timestamps(idx, tot, nat, 0.0)
        self.assertAlmostEqual(ts[0], 0.0); self.assertAlmostEqual(ts[1], 0.125, delta=0.03)
        self.assertLess(ts[-1], 5.0)
        sheet = core.render_contact_sheet(f, ts, cols, rows, 0, 0, "fit", 0, True, "index_timestamp", 160)
        self.assertGreater(sheet.shape[2], sheet.shape[1])

    def test_retimed_frames_keep_original_labels(self):
        # 5 s clip interpolated to 240 frames, tagged as 60 fps: labels must still span 5 s
        frames = torch.rand(240, 16, 16, 3)
        total, _ = core.probe_source(None, frames, 60.0)
        cols, rows, n = core.plan_layout(total, 5.0, 8.0)
        f, idx, tot, nat = core.sample_images(frames, n, "uniform", 60.0, "interval")
        ts = core.resolve_timestamps(idx, tot, nat, 5.0)
        self.assertEqual(n, 40)
        self.assertAlmostEqual(ts[8], 1.0, delta=0.02)
        self.assertLess(ts[-1], 5.0)

    def test_plan_caps(self):
        self.assertEqual(core.plan_layout(24, 100.0, 8.0)[2], 24)       # <= frames
        c, r, n = core.plan_layout(100000, 10000.0, 8.0)
        self.assertLessEqual(c, 32); self.assertLessEqual(r, 32)

    def test_partial_last_row_and_sizes(self):
        frames = torch.rand(10, 32, 48, 3)
        sheet = core.render_contact_sheet(frames, list(range(10)), 4, 3, 0, 0, "fit", 0, False, "none", 100)
        self.assertEqual(tuple(sheet.shape), (1, 201, 400, 3))
        sheet = core.render_contact_sheet(frames, list(range(10)), 4, 3, 400, 0, "fit", 0, False, "none", 0)
        self.assertEqual(sheet.shape[2], 400)

    def test_probe_file_video_and_reference(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "v.mp4"); make_video(p, 48, 24)
            v = FakeVideo(p, 24)
            n, dur = core.probe_source(v, None, 24.0)
            self.assertEqual(n, 48); self.assertAlmostEqual(dur, 2.0, delta=0.1)
            self.assertAlmostEqual(core.video_duration(v), 2.0, delta=0.1)


unittest.main()
