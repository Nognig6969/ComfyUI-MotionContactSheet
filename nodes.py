"""ComfyUI node: Motion Contact Sheet (XL)."""

from __future__ import annotations

from comfy_api.latest import ComfyExtension, io

from .core import (
    MAX_GRID_SIZE,
    _format_time,
    plan_layout,
    probe_source,
    render_contact_sheet,
    resolve_timestamps,
    sample_images,
    sample_video,
    video_duration,
)

PROMPT_HINT = (
    "The reference image is a chronological motion storyboard. Read its frames from left to right, "
    "then top to bottom. Follow the depicted motion and camera progression, but do not reproduce "
    "frame numbers, timestamps, film borders, perforations, or other storyboard markings."
)


class MotionContactSheetXL(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MotionContactSheetXL",
            display_name="Motion Contact Sheet (XL)",
            category="video/contact sheet",
            description=(
                "Sample N x N frames from a VIDEO or IMAGE batch and arrange them as an "
                "annotated, filmstrip-style motion storyboard. Supports grids up to "
                f"{MAX_GRID_SIZE}x{MAX_GRID_SIZE} and timestamps pinned to the original clip duration."
            ),
            inputs=[
                io.Int.Input(
                    "grid_size",
                    default=3,
                    min=0,
                    max=MAX_GRID_SIZE,
                    step=1,
                    tooltip=(
                        "Sample exactly grid_size squared frames in chronological row-major order. "
                        "0 = auto: the frame count comes from sample_fps x clip duration and the grid "
                        "can be rectangular."
                    ),
                ),
                io.Combo.Input(
                    "sampling",
                    options=["uniform", "motion_weighted"],
                    default="uniform",
                    tooltip="Uniform preserves timing. Motion weighted allocates more panels to intervals with stronger visual change.",
                ),
                io.Int.Input(
                    "width",
                    default=0,
                    min=0,
                    max=16384,
                    step=1,
                    tooltip="Final sheet width. 0 derives it from the source aspect and the height setting.",
                ),
                io.Int.Input(
                    "height",
                    default=0,
                    min=0,
                    max=16384,
                    step=1,
                    tooltip="Final sheet height. 0 derives it from the source aspect and the width setting.",
                ),
                io.Combo.Input(
                    "resize_mode",
                    options=["fit", "crop", "stretch"],
                    default="fit",
                    tooltip="How each sampled frame is fitted into its panel.",
                ),
                io.Int.Input(
                    "gap",
                    default=0,
                    min=0,
                    max=64,
                    step=1,
                    tooltip="Black pixels between adjacent panels.",
                ),
                io.Boolean.Input(
                    "film_border",
                    default=True,
                    tooltip="Wrap every panel in film rails and perforations. Labels are printed on the lower rail.",
                ),
                io.Combo.Input(
                    "annotation",
                    options=["none", "index", "timestamp", "index_timestamp"],
                    default="index",
                    tooltip="Optional chronological label. Without the film border, labels use a narrow caption rail outside the frame.",
                ),
                io.Float.Input(
                    "image_frame_rate",
                    default=24.0,
                    min=0.01,
                    max=1000.0,
                    step=0.01,
                    tooltip="Playback rate of an IMAGE batch. Only used to derive its duration when original_duration is 0 and no reference_video is connected.",
                ),
                io.Float.Input(
                    "original_duration",
                    default=0.0,
                    min=0.0,
                    max=100000.0,
                    step=0.01,
                    tooltip=(
                        "Length in seconds of the ORIGINAL clip. When above 0, timestamps are pinned to this "
                        "duration and ignore image_frame_rate and any frame-rate change/interpolation, so labels "
                        "always match the original timeline. 0 = take it from reference_video if connected, "
                        "otherwise derive it from the frame rate."
                    ),
                ),
                io.Float.Input(
                    "sample_fps",
                    default=2.0,
                    min=0.001,
                    max=1000.0,
                    step=0.001,
                    tooltip=(
                        "Only used when grid_size is 0 (auto). Panels per second of ORIGINAL video: "
                        "a 5 s clip at 8 gives 40 panels, laid out as the squarest grid that fits "
                        "(8 x 5). Capped at 32 x 32 and at the number of frames."
                    ),
                ),
                io.Int.Input(
                    "min_cell_width",
                    default=160,
                    min=0,
                    max=4096,
                    step=1,
                    tooltip=(
                        "Only used when width and height are both 0: enlarges the sheet so each panel is at "
                        "least this wide, which keeps big grids readable. 0 keeps the source resolution."
                    ),
                ),
                io.Video.Input(
                    "video",
                    optional=True,
                    tooltip="A loaded ComfyUI VIDEO. Connect either video or frames, not both.",
                ),
                io.Video.Input(
                    "reference_video",
                    optional=True,
                    tooltip=(
                        "Optional ORIGINAL video, used only to read its duration for the timestamps "
                        "(when original_duration is 0). Handy when frames/video were re-timed or interpolated."
                    ),
                ),
                io.Image.Input(
                    "frames",
                    optional=True,
                    tooltip="An already decoded video frame batch. Connect either frames or video, not both.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="contact_sheet"),
                io.Image.Output(display_name="sampled_frames"),
                io.String.Output(display_name="prompt_hint"),
                io.String.Output(display_name="timestamps"),
            ],
        )

    @classmethod
    def execute(
        cls,
        grid_size: int,
        sampling: str,
        width: int,
        height: int,
        resize_mode: str,
        gap: int,
        film_border: bool,
        annotation: str,
        image_frame_rate: float,
        original_duration: float = 0.0,
        sample_fps: float = 2.0,
        min_cell_width: int = 160,
        video=None,
        reference_video=None,
        frames=None,
    ) -> io.NodeOutput:
        if (video is None) == (frames is None):
            raise ValueError("Connect exactly one source: video or frames")
        total_frames, natural_duration = probe_source(video, frames, image_frame_rate)
        duration = float(original_duration)
        if duration <= 0.0 and reference_video is not None:
            duration = video_duration(reference_video)
        grid_size = int(grid_size)
        effective = duration if duration > 0.0 else natural_duration
        if grid_size > 0:
            columns = rows = grid_size
            sample_count, spread = grid_size**2, "endpoints"
        else:
            columns, rows, sample_count = plan_layout(
                total_frames, effective, float(sample_fps)
            )
            spread = "interval"
        if video is not None:
            sampled, indices, total, natural = sample_video(
                video, sample_count, sampling, spread
            )
        else:
            sampled, indices, total, natural = sample_images(
                frames, sample_count, sampling, image_frame_rate, spread
            )
        timestamps = resolve_timestamps(indices, total, natural, duration)
        sheet = render_contact_sheet(
            sampled,
            timestamps,
            columns,
            rows,
            int(width),
            int(height),
            resize_mode,
            int(gap),
            bool(film_border),
            annotation,
            int(min_cell_width),
        )
        stamp_text = f"grid {columns}x{rows} ({len(indices)} panels), duration {duration if duration > 0 else natural_duration:.2f}s\n" + "\n".join(
            f"{i + 1:02d}: frame {idx} @ {_format_time(t)}"
            for i, (idx, t) in enumerate(zip(indices, timestamps))
        )
        return io.NodeOutput(sheet, sampled, PROMPT_HINT, stamp_text)


class MotionContactSheetExtension(ComfyExtension):
    async def get_node_list(self):
        return [MotionContactSheetXL]


async def comfy_entrypoint() -> MotionContactSheetExtension:
    return MotionContactSheetExtension()
