"""Sampling, timestamp resolution and rendering for the motion contact sheet.

Everything here is independent of ComfyUI's node schema so it can be tested
with plain torch tensors.
"""

from __future__ import annotations

import bisect
import math
from functools import lru_cache

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

try:  # only needed to recognise in-memory VIDEO objects
    from comfy_api.latest import InputImpl
except Exception:  # pragma: no cover - lets the core be imported outside ComfyUI
    InputImpl = None

MAX_GRID_SIZE = 32


# --------------------------------------------------------------------------
# Frame index selection
# --------------------------------------------------------------------------
def uniform_frame_indices(frame_count: int, sample_count: int) -> list[int]:
    if frame_count < 1:
        raise ValueError("A motion contact sheet requires at least one frame")
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if sample_count == 1:
        return [frame_count // 2]
    return [
        round(i * (frame_count - 1) / (sample_count - 1)) for i in range(sample_count)
    ]


def motion_weighted_frame_indices(motion: list[float], sample_count: int) -> list[int]:
    frame_count = len(motion) + 1
    if sample_count == 1:
        return [frame_count // 2]
    if frame_count == 1 or not any(value > 0.0 for value in motion):
        return uniform_frame_indices(frame_count, sample_count)

    mean_motion = sum(motion) / len(motion)
    baseline = max(mean_motion * 0.15, 1e-6)
    cumulative = [0.0]
    for value in motion:
        cumulative.append(cumulative[-1] + max(value, 0.0) + baseline)

    indices = []
    total = cumulative[-1]
    for i in range(sample_count):
        target = total * i / max(sample_count - 1, 1)
        index = bisect.bisect_left(cumulative, target)
        if 0 < index < frame_count and abs(cumulative[index - 1] - target) <= abs(
            cumulative[index] - target
        ):
            index -= 1
        indices.append(min(index, frame_count - 1))
    indices[0] = 0
    indices[-1] = frame_count - 1
    return indices


def _motion_scores(frames: torch.Tensor) -> list[float]:
    frame_count, height, width, _ = frames.shape
    if frame_count < 2:
        return []
    stride_y = max(1, height // 64)
    stride_x = max(1, width // 64)
    previous = None
    scores = []
    for start in range(0, frame_count, 128):
        chunk = frames[start : start + 128, ::stride_y, ::stride_x, :3].float()
        gray = chunk[..., 0] * 0.299 + chunk[..., 1] * 0.587 + chunk[..., 2] * 0.114
        if previous is not None:
            scores.append(float((gray[0] - previous).abs().mean().item()))
        if gray.shape[0] > 1:
            scores.extend(
                (gray[1:] - gray[:-1]).abs().mean(dim=(1, 2)).detach().cpu().tolist()
            )
        previous = gray[-1]
    return scores


# --------------------------------------------------------------------------
# Streamed (file-backed) video decoding
# --------------------------------------------------------------------------
def _iter_video_frames(video):
    import av

    source = video.get_stream_source()
    start_time, duration = video.get_active_trim_window()
    end_time = start_time + duration if duration > 0.0 else None
    with av.open(source, mode="r") as container:
        if not container.streams.video:
            raise ValueError("The VIDEO input contains no video stream")
        stream = container.streams.video[0]
        if start_time > 0.0:
            container.seek(max(0, int(start_time / stream.time_base)), stream=stream)
        fallback_index = 0
        for frame in container.decode(stream):
            timestamp = (
                float(frame.pts * stream.time_base) if frame.pts is not None else None
            )
            if timestamp is not None and timestamp + 1e-9 < start_time:
                continue
            if timestamp is not None and end_time is not None and timestamp >= end_time:
                break
            relative_time = (
                max(0.0, timestamp - start_time)
                if timestamp is not None
                else fallback_index / float(video.get_frame_rate())
            )
            yield frame, relative_time
            fallback_index += 1


def _frame_to_tensor(frame) -> torch.Tensor:
    array = frame.to_ndarray(format="rgb24")
    return torch.from_numpy(array.copy()).float().div_(255.0)


def _decode_video_indices(video, indices: list[int]):
    wanted = set(indices)
    decoded = {}
    last_frame = None
    last_time = 0.0
    max_index = max(indices)
    for index, (frame, timestamp) in enumerate(_iter_video_frames(video)):
        last_frame = frame
        last_time = timestamp
        if index in wanted:
            decoded[index] = (_frame_to_tensor(frame), timestamp)
        if index >= max_index:
            break
    if last_frame is None:
        raise ValueError("The VIDEO input contains no decodable frames")

    tail = (_frame_to_tensor(last_frame), last_time)
    available = sorted(decoded)
    output_frames, output_times = [], []
    for index in indices:
        if index in decoded:
            image, timestamp = decoded[index]
        elif not available or index > available[-1]:
            image, timestamp = tail
        else:
            nearest = min(available, key=lambda candidate: abs(candidate - index))
            image, timestamp = decoded[nearest]
        output_frames.append(image)
        output_times.append(timestamp)
    return torch.stack(output_frames), output_times


def _video_motion_scores(video) -> list[float]:
    previous = None
    scores = []
    for frame, _ in _iter_video_frames(video):
        thumb_width = min(64, frame.width)
        thumb_height = max(1, round(frame.height * thumb_width / frame.width))
        thumb = frame.reformat(width=thumb_width, height=thumb_height, format="gray")
        gray = torch.from_numpy(thumb.to_ndarray().copy()).float()
        if previous is not None:
            scores.append(float((gray - previous).abs().mean().item()))
        previous = gray
    if previous is None:
        raise ValueError("The VIDEO input contains no decodable frames")
    return scores


# --------------------------------------------------------------------------
# Source probing (auto grid size / auto duration)
# --------------------------------------------------------------------------
def video_duration(video) -> float:
    """Duration in seconds of any ComfyUI VIDEO."""
    try:
        value = float(video.get_duration())
        if value > 0.0:
            return value
    except Exception:
        pass
    return int(video.get_frame_count()) / float(video.get_frame_rate())


def probe_source(video, frames, image_frame_rate: float):
    """Cheap (frame_count, duration_seconds) of the connected source."""
    if video is not None:
        if InputImpl is not None and isinstance(video, InputImpl.VideoFromComponents):
            comps = video.get_components()
            count = int(comps.images.shape[0])
            return count, count / float(comps.frame_rate)
        return int(video.get_frame_count()), video_duration(video)
    if not torch.is_tensor(frames) or frames.ndim != 4:
        raise ValueError("Expected IMAGE frames shaped [frames,height,width,channels]")
    count = int(frames.shape[0])
    return count, count / float(image_frame_rate)


def interval_frame_indices(frame_count: int, sample_count: int) -> list[int]:
    """Evenly spaced frames on a time grid: panel k sits at k/sample_count of the clip."""
    if frame_count < 1 or sample_count < 1:
        raise ValueError("frame_count and sample_count must be positive")
    return [min(frame_count - 1, round(k * frame_count / sample_count)) for k in range(sample_count)]


def choose_layout(panels: int) -> tuple[int, int]:
    """(columns, rows) holding at least ``panels`` cells, as close to square as possible.

    An exact near-square factor pair (40 -> 8 x 5) is preferred; otherwise the
    smallest roughly square grid that fits (41 -> 7 x 6).
    """
    panels = max(1, int(panels))
    for rows in range(math.isqrt(panels), 0, -1):
        if panels % rows == 0 and (panels // rows) <= 2 * rows:
            return panels // rows, rows
    columns = math.ceil(math.sqrt(panels))
    return columns, math.ceil(panels / columns)


def plan_layout(total_frames: int, duration: float, sample_fps: float):
    """Auto mode: (columns, rows, sample_count) for ``sample_fps`` panels per second."""
    limit = min(max(1, int(total_frames)), MAX_GRID_SIZE**2)
    wanted = max(1, round(duration * sample_fps)) if duration > 0 else limit
    columns, rows = choose_layout(min(wanted, limit))
    while columns * rows > MAX_GRID_SIZE**2 or columns > MAX_GRID_SIZE:
        columns, rows = choose_layout(MAX_GRID_SIZE**2)
    return columns, rows, min(columns * rows, limit)


# --------------------------------------------------------------------------
# Public sampling API
# --------------------------------------------------------------------------
def resolve_timestamps(
    indices: list[int],
    total_frames: int,
    natural_times: list[float],
    original_duration: float,
) -> list[float]:
    """Return the label time for every sampled frame.

    original_duration <= 0 keeps the "natural" times (embedded video times or
    index / frame_rate).  Otherwise the clip is treated as spanning exactly
    ``original_duration`` seconds, however many frames it has now or whatever
    frame rate it is tagged with, so frame ``i`` of ``N`` is labelled
    ``i * original_duration / N`` (the same convention as index / fps).
    """
    if original_duration is None or original_duration <= 0.0:
        return list(natural_times)
    total = max(1, int(total_frames))
    return [index * float(original_duration) / total for index in indices]


def sample_images(
    frames: torch.Tensor,
    sample_count: int,
    sampling: str,
    frame_rate: float,
    spread: str = "endpoints",
):
    """Returns (frames, indices, total_frames, natural_times)."""
    if not torch.is_tensor(frames) or frames.ndim != 4 or frames.shape[-1] < 3:
        shape = (
            tuple(frames.shape) if hasattr(frames, "shape") else type(frames).__name__
        )
        raise ValueError(
            f"Expected IMAGE frames shaped [frames,height,width,channels], got {shape}"
        )
    if frames.shape[0] < 1:
        raise ValueError("The IMAGE input contains no frames")
    if frame_rate <= 0.0:
        raise ValueError("image_frame_rate must be positive")
    total = int(frames.shape[0])
    if sampling == "motion_weighted":
        indices = motion_weighted_frame_indices(_motion_scores(frames), sample_count)
    elif spread == "interval":
        indices = interval_frame_indices(total, sample_count)
    else:
        indices = uniform_frame_indices(total, sample_count)
    return (
        frames[indices, ..., :3],
        indices,
        total,
        [index / frame_rate for index in indices],
    )


def sample_video(video, sample_count: int, sampling: str, spread: str = "endpoints"):
    """Returns (frames, indices, total_frames, natural_times)."""
    if InputImpl is not None and isinstance(video, InputImpl.VideoFromComponents):
        components = video.get_components()
        return sample_images(
            components.images,
            sample_count,
            sampling,
            float(components.frame_rate),
            spread,
        )
    if sampling == "motion_weighted":
        scores = _video_motion_scores(video)
        total = len(scores) + 1
        indices = motion_weighted_frame_indices(scores, sample_count)
    else:
        total = int(video.get_frame_count())
        if spread == "interval":
            indices = interval_frame_indices(total, sample_count)
        else:
            indices = uniform_frame_indices(total, sample_count)
    frames, times = _decode_video_indices(video, indices)
    return frames, indices, total, times


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def _resolved_size(source_width, source_height, width, height):
    if width == 0 and height == 0:
        return source_width, source_height
    if width == 0:
        return max(1, round(height * source_width / source_height)), height
    if height == 0:
        return width, max(1, round(width * source_height / source_width))
    return width, height


def resolve_sheet_size(
    source_width: int,
    source_height: int,
    width: int,
    height: int,
    columns: int,
    rows: int,
    gap: int,
    min_cell_width: int,
) -> tuple[int, int]:
    """Sheet size. Panels always keep the source aspect ratio unless both sizes are given."""
    if width > 0 and height > 0:
        return width, height
    aspect = source_height / source_width
    if width > 0:
        cell_w = max(1.0, (width - gap * (columns - 1)) / columns)
        return width, round(rows * max(1, round(cell_w * aspect)) + gap * (rows - 1))
    if height > 0:
        cell_h = max(1.0, (height - gap * (rows - 1)) / rows)
        cell_w = max(1, round(cell_h / aspect))
        return columns * cell_w + gap * (columns - 1), height
    cell_w = max(int(min_cell_width), source_width // columns, 1)
    cell_h = max(1, round(cell_w * aspect))
    return (
        columns * cell_w + gap * (columns - 1),
        rows * cell_h + gap * (rows - 1),
    )


def _split_extent(total: int, count: int, gap: int):
    usable = total - gap * (count - 1)
    if usable < count:
        raise ValueError(
            f"Output extent {total} is too small for {count} cells and gap={gap}"
        )
    base, remainder = divmod(usable, count)
    segments, position = [], 0
    for index in range(count):
        size = base + (1 if index < remainder else 0)
        segments.append((position, size))
        position += size + gap
    return segments


def _fit_image(image, width, height, resize_mode):
    if resize_mode == "stretch":
        return image.resize((width, height), Image.Resampling.LANCZOS)
    if resize_mode == "crop":
        return ImageOps.fit(
            image, (width, height), Image.Resampling.LANCZOS, centering=(0.5, 0.5)
        )
    resized = ImageOps.contain(image, (width, height), Image.Resampling.LANCZOS)
    output = Image.new("RGB", (width, height), (8, 8, 8))
    output.paste(
        resized, ((width - resized.width) // 2, (height - resized.height) // 2)
    )
    return output


def _format_time(seconds: float) -> str:
    return f"{seconds:.2f}s"


def _label(annotation: str, index: int, timestamp: float) -> str:
    if annotation == "index":
        return f"{index + 1:02d}"
    if annotation == "timestamp":
        return _format_time(timestamp)
    if annotation == "index_timestamp":
        return f"{index + 1:02d} | {_format_time(timestamp)}"
    return ""


@lru_cache(maxsize=32)
def _font(size: int):
    return ImageFont.load_default(size=size)


def _draw_centered_label(draw, label, y0, width, height):
    if not label or height < 6:
        return None
    font = _font(max(8, round(height * 0.62)))
    bounds = draw.textbbox((0, 0), label, font=font)
    text_width = bounds[2] - bounds[0]
    text_height = bounds[3] - bounds[1]
    x = max(1, (width - text_width) // 2)
    y = y0 + max(0, (height - text_height) // 2 - bounds[1])
    draw.text((x, y), label, fill=(235, 235, 220), font=font)
    return x, x + text_width


def _draw_perforations(draw, width, tile_height, rail_height, label_span):
    hole_height = max(2, rail_height // 3)
    hole_width = max(4, round(hole_height * 1.7))
    step = max(hole_width + 3, round(hole_width * 1.8))
    top_y = max(1, (rail_height - hole_height) // 2)
    bottom_y = tile_height - rail_height + top_y
    for x in range(3, width - hole_width, step):
        draw.rounded_rectangle(
            (x, top_y, x + hole_width, top_y + hole_height),
            radius=max(1, hole_height // 4),
            fill=(224, 216, 188),
        )
        if (
            label_span is None
            or x + hole_width < label_span[0] - 4
            or x > label_span[1] + 4
        ):
            draw.rounded_rectangle(
                (x, bottom_y, x + hole_width, bottom_y + hole_height),
                radius=max(1, hole_height // 4),
                fill=(224, 216, 188),
            )


def _render_tile(frame, width, height, resize_mode, film_border, label):
    if width < 8 or height < 8:
        raise ValueError(
            f"Contact-sheet cells must be at least 8x8 pixels, got {width}x{height}. "
            "Use a larger width/height or a smaller grid_size."
        )
    array = frame.detach().float().clamp(0.0, 1.0).mul(255.0).byte().cpu().numpy()
    source = Image.fromarray(array[..., :3])
    tile = Image.new("RGB", (width, height), (9, 9, 8))
    draw = ImageDraw.Draw(tile)

    if film_border:
        rail_height = min(max(10, round(height * 0.09)), max(10, height // 4))
        viewport_height = height - rail_height * 2
        if viewport_height < 8:
            raise ValueError(
                f"Cell height {height} is too small for the film border; "
                "disable film_border, enlarge the sheet, or lower grid_size"
            )
        tile.paste(
            _fit_image(source, width, viewport_height, resize_mode), (0, rail_height)
        )
        draw.rectangle(
            (0, rail_height, width - 1, height - rail_height - 1), outline=(65, 62, 52)
        )
        label_span = _draw_centered_label(
            draw, label, height - rail_height, width, rail_height
        )
        _draw_perforations(draw, width, height, rail_height, label_span)
    elif label:
        caption_height = min(max(10, round(height * 0.08)), max(10, height // 4))
        tile.paste(
            _fit_image(source, width, height - caption_height, resize_mode), (0, 0)
        )
        _draw_centered_label(draw, label, height - caption_height, width, caption_height)
    else:
        tile.paste(_fit_image(source, width, height, resize_mode), (0, 0))
    return tile


def render_contact_sheet(
    frames: torch.Tensor,
    timestamps: list[float],
    columns: int,
    rows: int,
    width: int,
    height: int,
    resize_mode: str,
    gap: int,
    film_border: bool,
    annotation: str,
    min_cell_width: int = 0,
) -> torch.Tensor:
    count = int(frames.shape[0])
    if count != len(timestamps) or count > columns * rows or count < 1:
        raise ValueError(
            f"A {columns}x{rows} contact sheet holds up to {columns * rows} frames; "
            f"got {count} frames and {len(timestamps)} timestamps"
        )
    source_height, source_width = int(frames.shape[1]), int(frames.shape[2])
    width, height = resolve_sheet_size(
        source_width, source_height, width, height, columns, rows, gap, min_cell_width
    )
    col_segments = _split_extent(width, columns, gap)
    row_segments = _split_extent(height, rows, gap)
    sheet = Image.new("RGB", (width, height), (5, 5, 5))
    for index, frame in enumerate(frames):
        row, column = divmod(index, columns)
        x, tile_width = col_segments[column]
        y, tile_height = row_segments[row]
        tile = _render_tile(
            frame,
            tile_width,
            tile_height,
            resize_mode,
            film_border,
            _label(annotation, index, timestamps[index]),
        )
        sheet.paste(tile, (x, y))
    array = np.array(sheet, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)
