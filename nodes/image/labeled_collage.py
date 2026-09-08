"""Create labeled horizontal or vertical image collages."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import numpy as np
import torch
from comfy_api.latest import io
from PIL import Image, ImageColor, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_INPUT_NAMES = [f"image_{index}" for index in range(1, 33)]
FONT_EXTENSIONS = {".ttf", ".otf", ".ttc", ".otc"}
LABEL_POSITIONS = [
    f"{vertical} {horizontal}"
    for vertical in ("top", "center", "bottom")
    for horizontal in ("left", "center", "right")
]
AUTO_FONT = "Auto (CJK preferred)"


def _resampling_lanczos():
    return getattr(Image, "Resampling", Image).LANCZOS


def _parse_color(value: str) -> tuple[int, int, int, int]:
    rgba_float = re.fullmatch(
        r"rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(0(?:\.\d+)?|1(?:\.0+)?)\s*\)",
        value,
        flags=re.IGNORECASE,
    )
    if rgba_float:
        red, green, blue = (int(rgba_float.group(index)) for index in range(1, 4))
        alpha = round(float(rgba_float.group(4)) * 255)
        if any(channel < 0 or channel > 255 for channel in (red, green, blue)):
            raise ValueError(f"Invalid color value: {value!r}.")
        return red, green, blue, alpha
    try:
        rgba = ImageColor.getcolor(value, "RGBA")
    except ValueError as exc:
        raise ValueError(f"Invalid color value: {value!r}.") from exc
    return tuple(int(channel) for channel in rgba)


def _tensor_items(images: dict[str, torch.Tensor] | None) -> list[torch.Tensor]:
    items: list[torch.Tensor] = []
    images = images or {}
    for name in IMAGE_INPUT_NAMES:
        batch = images.get(name)
        if batch is None:
            continue
        if not torch.is_tensor(batch) or batch.ndim != 4 or batch.shape[-1] < 3:
            raise ValueError(f"{name} must be an RGB IMAGE batch [B,H,W,C].")
        channel_count = 4 if batch.shape[-1] >= 4 else 3
        items.extend(batch[..., :channel_count].detach().to(device="cpu", dtype=torch.float32))
    if not items:
        raise ValueError("Connect at least one image to Labeled Image Collage Ψ.")
    return items


def _to_pil(item: torch.Tensor) -> Image.Image:
    array = np.clip(np.rint(item.numpy() * 255.0), 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGBA" if array.shape[-1] == 4 else "RGB")


def _font_candidates() -> Iterable[Path]:
    for directory in (
        PROJECT_ROOT / "assets" / "fonts",
        PROJECT_ROOT / "incoming",
        PROJECT_ROOT / "temp",
    ):
        if not directory.is_dir():
            continue
        paths = [path for path in directory.iterdir() if path.suffix.lower() in FONT_EXTENSIONS]
        paths.sort(
            key=lambda path: (
                "notosans" not in path.name.lower(),
                "noto" not in path.name.lower(),
                path.name.lower(),
            )
        )
        yield from paths

    windows_fonts = Path("C:/Windows/Fonts")
    for name in ("msjh.ttc", "msjh.ttf", "mingliu.ttc", "malgun.ttf", "msgothic.ttc"):
        path = windows_fonts / name
        if path.is_file():
            yield path


def _display_font_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _font_faces(path: Path) -> list[tuple[str, Path, int]]:
    faces: list[tuple[str, Path, int]] = []
    for index in range(32):
        try:
            font = ImageFont.truetype(str(path), 16, index=index)
        except (OSError, ValueError):
            break
        family, style = font.getname()
        option = f"{_display_font_path(path)} | {family} {style}".strip()
        faces.append((option, path, index))
    return faces


def font_records(paths: Iterable[Path] | None = None) -> list[tuple[str, Path, int]]:
    records: list[tuple[str, Path, int]] = []
    seen: set[tuple[str, int]] = set()
    for path in paths if paths is not None else _font_candidates():
        for record in _font_faces(path):
            key = (str(record[1].resolve()).lower(), record[2])
            if key not in seen:
                records.append(record)
                seen.add(key)
    return records


def font_options() -> list[str]:
    return [AUTO_FONT, *(record[0] for record in font_records())]


def _font_priority(record: tuple[str, Path, int]) -> tuple[bool, bool, bool, str]:
    description = record[0].lower()
    is_traditional_cjk = any(
        marker in description
        for marker in (" cjk tc", "traditional", "jhenghei", "mingliu")
    )
    return (
        not is_traditional_cjk,
        "sans" not in description,
        "noto" not in description,
        description,
    )


def load_font(
    font_name: str,
    font_path: str,
    font_size: int,
) -> ImageFont.ImageFont:
    """Load a listed font face or an automatically selected CJK-friendly face."""

    if font_path.strip():
        requested = Path(font_path.strip()).expanduser()
        if not requested.is_absolute():
            requested = PROJECT_ROOT / requested
        if not requested.is_file():
            raise FileNotFoundError(f"Font file not found: {requested}")
        records = font_records([requested])
        explicit = True
    else:
        records = font_records()
        explicit = False

    if explicit:
        selected = min(records, key=_font_priority) if records else None
    elif font_name == AUTO_FONT:
        selected = min(records, key=_font_priority) if records else None
    else:
        selected = next((record for record in records if record[0] == font_name), None)
        if selected is None:
            raise ValueError(
                f"Selected font is no longer available: {font_name!r}. "
                "Choose another listed font or Auto."
            )

    if selected is not None:
        try:
            return ImageFont.truetype(str(selected[1]), int(font_size), index=selected[2])
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Unable to load {selected[0]!r}: {exc}") from exc
    if explicit:
        raise RuntimeError(f"Unable to read any font faces from: {requested}")
    try:
        return ImageFont.truetype("DejaVuSans.ttf", int(font_size))
    except OSError:
        return ImageFont.load_default()


def _fit_image(image: Image.Image, width: int, height: int, background: tuple[int, ...]) -> Image.Image:
    scale = min(width / image.width, height / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        _resampling_lanczos(),
    )
    mode = "RGBA" if background[3] < 255 or image.mode == "RGBA" else "RGB"
    fill = background if mode == "RGBA" else background[:3]
    cell = Image.new(mode, (width, height), fill)
    if resized.mode != mode:
        resized = resized.convert(mode)
    cell.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return cell


def prepare_images(
    images: list[Image.Image],
    layout: str,
    size_mode: str,
    cell_width: int,
    cell_height: int,
    cross_axis_size: int,
    background: tuple[int, ...],
) -> list[Image.Image]:
    if size_mode == "uniform cells (fit)":
        return [_fit_image(image, cell_width, cell_height, background) for image in images]
    if size_mode != "match cross-axis":
        raise ValueError(f"Unsupported size mode: {size_mode!r}.")

    prepared: list[Image.Image] = []
    for image in images:
        if layout == "horizontal":
            scale = cross_axis_size / image.height
            size = (max(1, round(image.width * scale)), cross_axis_size)
        elif layout == "vertical":
            scale = cross_axis_size / image.width
            size = (cross_axis_size, max(1, round(image.height * scale)))
        else:
            raise ValueError(f"Unsupported layout: {layout!r}.")
        prepared.append(image.resize(size, _resampling_lanczos()))
    return prepared


def _wrap_horizontal_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    stroke_width: int,
) -> str:
    wrapped: list[str] = []
    for source_line in text.split("\n"):
        if not source_line:
            wrapped.append("")
            continue
        current = ""
        for character in source_line:
            candidate = current + character
            bounds = draw.textbbox((0, 0), candidate, font=font, stroke_width=stroke_width)
            if current and bounds[2] - bounds[0] > max_width:
                wrapped.append(current)
                current = character
            else:
                current = candidate
        wrapped.append(current)
    return "\n".join(wrapped)


def _vertical_text(text: str) -> str:
    lines = text.split("\n")
    return "\n\n".join("\n".join(line) for line in lines)


def _measure_label(
    image: Image.Image,
    text: str,
    font: ImageFont.ImageFont,
    text_direction: str,
    label_padding: int,
    label_margin: int,
    outline_width: int,
) -> tuple[str, int, tuple[int, int, int, int], int, int]:
    measurement = ImageDraw.Draw(Image.new("L", (1, 1)))
    max_text_width = max(1, image.width - 2 * (label_margin + label_padding + outline_width))
    if text_direction == "horizontal":
        rendered_text = _wrap_horizontal_text(measurement, text, font, max_text_width, outline_width)
    elif text_direction == "vertical":
        rendered_text = _vertical_text(text)
    else:
        raise ValueError(f"Unsupported text direction: {text_direction!r}.")

    line_spacing = max(2, int(getattr(font, "size", 12) * 0.15))
    bounds = measurement.multiline_textbbox(
        (0, 0),
        rendered_text,
        font=font,
        spacing=line_spacing,
        align="center",
        stroke_width=outline_width,
    )
    text_width = max(1, bounds[2] - bounds[0])
    text_height = max(1, bounds[3] - bounds[1])
    return (
        rendered_text,
        line_spacing,
        bounds,
        text_width + 2 * label_padding,
        text_height + 2 * label_padding,
    )


def label_extension_size(
    image: Image.Image,
    text: str,
    font: ImageFont.ImageFont,
    text_direction: str,
    label_placement: str,
    label_padding: int,
    label_margin: int,
    outline_width: int,
) -> int:
    if not text or label_placement == "overlay":
        return 0
    if label_placement not in ("extend top", "extend bottom", "extend left", "extend right"):
        raise ValueError(f"Unsupported label placement: {label_placement!r}.")
    _, _, _, tight_width, tight_height = _measure_label(
        image,
        text,
        font,
        text_direction,
        label_padding,
        label_margin,
        outline_width,
    )
    if label_placement in ("extend top", "extend bottom"):
        return tight_height + 2 * label_margin
    return tight_width + 2 * label_margin


def draw_label(
    image: Image.Image,
    text: str,
    font: ImageFont.ImageFont,
    text_direction: str,
    label_position: str,
    label_placement: str,
    font_color: tuple[int, int, int, int],
    label_background: tuple[int, int, int, int],
    label_background_extent: str,
    label_padding: int,
    label_margin: int,
    outline_color: tuple[int, int, int, int],
    outline_width: int,
    extension_size: int | None = None,
) -> Image.Image:
    if label_placement == "overlay" and not text:
        return image

    vertical, horizontal = label_position.split(" ", 1)
    if label_placement != "overlay":
        if label_placement not in ("extend top", "extend bottom", "extend left", "extend right"):
            raise ValueError(f"Unsupported label placement: {label_placement!r}.")
        required_size = label_extension_size(
            image,
            text,
            font,
            text_direction,
            label_placement,
            label_padding,
            label_margin,
            outline_width,
        )
        strip_size = required_size if extension_size is None else max(0, int(extension_size))
        if strip_size == 0:
            return image

        horizontal_strip = label_placement in ("extend top", "extend bottom")
        canvas_size = (
            (image.width, image.height + strip_size)
            if horizontal_strip
            else (image.width + strip_size, image.height)
        )
        mode = "RGBA" if image.mode == "RGBA" or label_background[3] < 255 else "RGB"
        fill = label_background if mode == "RGBA" else label_background[:3]
        canvas = Image.new(mode, canvas_size, fill)
        image_offset = {
            "extend top": (0, strip_size),
            "extend bottom": (0, 0),
            "extend left": (strip_size, 0),
            "extend right": (0, 0),
        }[label_placement]
        canvas.paste(image.convert(mode), image_offset)

        if text:
            rendered_text, line_spacing, bounds, tight_width, tight_height = _measure_label(
                image,
                text,
                font,
                text_direction,
                label_padding,
                label_margin,
                outline_width,
            )
            if label_placement == "extend top":
                strip_x, strip_y, strip_width, strip_height = 0, 0, image.width, strip_size
            elif label_placement == "extend bottom":
                strip_x, strip_y, strip_width, strip_height = 0, image.height, image.width, strip_size
            elif label_placement == "extend left":
                strip_x, strip_y, strip_width, strip_height = 0, 0, strip_size, image.height
            else:
                strip_x, strip_y, strip_width, strip_height = image.width, 0, strip_size, image.height

            if horizontal_strip:
                if horizontal == "left":
                    tight_x = strip_x + label_margin
                elif horizontal == "center":
                    tight_x = strip_x + (strip_width - tight_width) // 2
                else:
                    tight_x = strip_x + strip_width - label_margin - tight_width
                tight_y = strip_y + (strip_height - tight_height) // 2
            else:
                tight_x = strip_x + (strip_width - tight_width) // 2
                if vertical == "top":
                    tight_y = strip_y + label_margin
                elif vertical == "center":
                    tight_y = strip_y + (strip_height - tight_height) // 2
                else:
                    tight_y = strip_y + strip_height - label_margin - tight_height

            tight_x = max(strip_x, min(tight_x, strip_x + strip_width - tight_width))
            tight_y = max(strip_y, min(tight_y, strip_y + strip_height - tight_height))
            ImageDraw.Draw(canvas).multiline_text(
                (tight_x + label_padding - bounds[0], tight_y + label_padding - bounds[1]),
                rendered_text,
                font=font,
                fill=font_color,
                spacing=line_spacing,
                align="center",
                stroke_width=outline_width,
                stroke_fill=outline_color,
            )
        return canvas

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    rendered_text, line_spacing, bounds, tight_width, tight_height = _measure_label(
        image,
        text,
        font,
        text_direction,
        label_padding,
        label_margin,
        outline_width,
    )

    box_width = image.width if label_background_extent == "full width banner" else tight_width
    box_height = image.height if label_background_extent == "full height banner" else tight_height
    if label_background_extent not in ("text box", "full width banner", "full height banner"):
        raise ValueError(f"Unsupported label background extent: {label_background_extent!r}.")

    if horizontal == "left":
        tight_x = label_margin
    elif horizontal == "center":
        tight_x = (image.width - tight_width) // 2
    else:
        tight_x = image.width - label_margin - tight_width
    if vertical == "top":
        tight_y = label_margin
    elif vertical == "center":
        tight_y = (image.height - tight_height) // 2
    else:
        tight_y = image.height - label_margin - tight_height

    box_x = 0 if label_background_extent == "full width banner" else tight_x
    box_y = 0 if label_background_extent == "full height banner" else tight_y
    box_x = max(0, min(box_x, image.width - box_width))
    box_y = max(0, min(box_y, image.height - box_height))
    alpha = label_background[3]
    if alpha:
        radius = (
            0
            if label_background_extent != "text box"
            else min(label_padding, max(0, min(box_width, box_height) // 4))
        )
        draw.rounded_rectangle(
            (box_x, box_y, box_x + box_width, box_y + box_height),
            radius=radius,
            fill=(*label_background[:3], alpha),
        )

    text_x = max(0, min(tight_x, image.width - tight_width)) + label_padding - bounds[0]
    text_y = max(0, min(tight_y, image.height - tight_height)) + label_padding - bounds[1]
    draw.multiline_text(
        (text_x, text_y),
        rendered_text,
        font=font,
        fill=font_color,
        spacing=line_spacing,
        align="center",
        stroke_width=outline_width,
        stroke_fill=outline_color,
    )
    composited = Image.alpha_composite(image.convert("RGBA"), overlay)
    return composited if image.mode == "RGBA" else composited.convert("RGB")


def compose_collage(
    images: list[Image.Image],
    layout: str,
    spacing: int,
    outer_margin: int,
    background: tuple[int, ...],
    items_per_line: int = 32,
) -> Image.Image:
    if not images:
        raise ValueError("At least one prepared image is required.")
    items_per_line = int(items_per_line)
    if items_per_line < 1:
        raise ValueError("items_per_line must be at least 1.")

    mode = "RGBA" if background[3] < 255 or any(image.mode == "RGBA" for image in images) else "RGB"
    fill = background if mode == "RGBA" else background[:3]
    if layout == "horizontal":
        column_count = min(items_per_line, len(images))
        row_count = (len(images) + column_count - 1) // column_count
        positions = [(index // column_count, index % column_count) for index in range(len(images))]
    elif layout == "vertical":
        row_count = min(items_per_line, len(images))
        column_count = (len(images) + row_count - 1) // row_count
        positions = [(index % row_count, index // row_count) for index in range(len(images))]
    else:
        raise ValueError(f"Unsupported layout: {layout!r}.")

    column_widths = [0] * column_count
    row_heights = [0] * row_count
    for image, (row, column) in zip(images, positions):
        column_widths[column] = max(column_widths[column], image.width)
        row_heights[row] = max(row_heights[row], image.height)

    width = sum(column_widths) + spacing * (column_count - 1)
    height = sum(row_heights) + spacing * (row_count - 1)
    canvas = Image.new(mode, (width + 2 * outer_margin, height + 2 * outer_margin), fill)
    column_offsets = [outer_margin]
    row_offsets = [outer_margin]
    for column_width in column_widths[:-1]:
        column_offsets.append(column_offsets[-1] + column_width + spacing)
    for row_height in row_heights[:-1]:
        row_offsets.append(row_offsets[-1] + row_height + spacing)

    for image, (row, column) in zip(images, positions):
        x = column_offsets[column] + (column_widths[column] - image.width) // 2
        y = row_offsets[row] + (row_heights[row] - image.height) // 2
        canvas.paste(image.convert(mode), (x, y))
    return canvas


class LabeledImageCollage(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LabeledImageCollage_Sai",
            display_name="Labeled Image Collage Ψ",
            category="Sai/Image/Collage",
            description=(
                "Builds a horizontal or vertical collage from growing IMAGE inputs. "
                "Each line in labels maps to one image while all label styling is shared."
            ),
            search_aliases=["image collage", "labeled collage", "caption collage", "contact sheet"],
            inputs=[
                io.Autogrow.Input(
                    "images",
                    template=io.Autogrow.TemplateNames(
                        io.Image.Input("image"),
                        names=IMAGE_INPUT_NAMES,
                        min=1,
                    ),
                    tooltip="Connect images in label order; the final socket grows automatically.",
                ),
                io.String.Input(
                    "labels",
                    multiline=True,
                    default="",
                    dynamic_prompts=False,
                    tooltip="One label per flattened image. Blank or missing lines draw no label.",
                ),
                io.Combo.Input("layout", options=["horizontal", "vertical"], default="horizontal"),
                io.Int.Input(
                    "items_per_line",
                    default=32,
                    min=1,
                    max=32,
                    step=1,
                    tooltip=(
                        "Maximum columns before wrapping for horizontal layout, or maximum rows "
                        "before wrapping for vertical layout. Empty grid cells use background_color."
                    ),
                ),
                io.Combo.Input(
                    "size_mode",
                    options=["uniform cells (fit)", "match cross-axis"],
                    default="uniform cells (fit)",
                ),
                io.Int.Input("cell_width", default=512, min=16, max=8192, step=8),
                io.Int.Input("cell_height", default=512, min=16, max=8192, step=8),
                io.Int.Input(
                    "cross_axis_size",
                    default=512,
                    min=16,
                    max=8192,
                    step=8,
                    tooltip="Height for horizontal layout; width for vertical layout.",
                ),
                io.Int.Input("spacing", default=0, min=0, max=1024, step=1),
                io.Int.Input("outer_margin", default=0, min=0, max=1024, step=1),
                io.Color.Input("background_color", default="#000000"),
                io.Combo.Input(
                    "label_placement",
                    options=["overlay", "extend top", "extend bottom", "extend left", "extend right"],
                    default="overlay",
                    tooltip=(
                        "Overlay draws on the image. Extend modes add a shared-size label strip outside "
                        "every image so the original pixels remain unchanged."
                    ),
                ),
                io.Combo.Input("label_position", options=LABEL_POSITIONS, default="top center"),
                io.Combo.Input(
                    "text_direction",
                    options=["horizontal", "vertical"],
                    default="horizontal",
                ),
                io.Int.Input("font_size", default=48, min=4, max=1024, step=1),
                io.Color.Input("font_color", default="#ffffffff"),
                io.Color.Input("label_background_color", default="#00000000"),
                io.Combo.Input(
                    "label_background_extent",
                    options=["text box", "full width banner", "full height banner"],
                    default="text box",
                ),
                io.Int.Input("label_padding", default=8, min=0, max=256, step=1),
                io.Int.Input("label_margin", default=0, min=0, max=1024, step=1),
                io.Color.Input("outline_color", default="#000000ff", advanced=True),
                io.Int.Input("outline_width", default=2, min=0, max=64, step=1, advanced=True),
                io.Combo.Input(
                    "font_name",
                    options=font_options(),
                    default=AUTO_FONT,
                    tooltip="Discovered font file and face name. Refresh node definitions after adding fonts.",
                ),
                io.String.Input(
                    "font_path",
                    default="",
                    advanced=True,
                    tooltip=(
                        "Optional unlisted TTF/OTF/TTC/OTC path. Relative paths start at this node pack; "
                        "the Traditional Chinese face is selected automatically when available."
                    ),
                ),
            ],
            outputs=[io.Image.Output(display_name="collage")],
        )

    @classmethod
    def execute(
        cls,
        images: io.Autogrow.Type,
        labels: str,
        layout: str,
        items_per_line: int,
        size_mode: str,
        cell_width: int,
        cell_height: int,
        cross_axis_size: int,
        spacing: int,
        outer_margin: int,
        background_color: str,
        label_placement: str,
        label_position: str,
        text_direction: str,
        font_size: int,
        font_color: str,
        label_background_color: str,
        label_background_extent: str,
        label_padding: int,
        label_margin: int,
        outline_color: str,
        outline_width: int,
        font_name: str,
        font_path: str,
    ) -> io.NodeOutput:
        tensors = _tensor_items(images)
        pil_images = [_to_pil(item) for item in tensors]
        background = _parse_color(background_color)
        prepared = prepare_images(
            pil_images,
            layout,
            size_mode,
            int(cell_width),
            int(cell_height),
            int(cross_axis_size),
            background,
        )

        label_lines = labels.replace("\r\n", "\n").replace("\r", "\n").split("\n") if labels else []
        font = load_font(font_name, font_path, int(font_size))
        captions = [label_lines[index] if index < len(label_lines) else "" for index in range(len(prepared))]
        shared_extension_size = None
        if label_placement != "overlay":
            shared_extension_size = max(
                label_extension_size(
                    image,
                    caption,
                    font,
                    text_direction,
                    label_placement,
                    int(label_padding),
                    int(label_margin),
                    int(outline_width),
                )
                for image, caption in zip(prepared, captions)
            )
        styled = [
            draw_label(
                image,
                captions[index],
                font,
                text_direction,
                label_position,
                label_placement,
                _parse_color(font_color),
                _parse_color(label_background_color),
                label_background_extent,
                int(label_padding),
                int(label_margin),
                _parse_color(outline_color),
                int(outline_width),
                shared_extension_size,
            )
            for index, image in enumerate(prepared)
        ]
        collage = compose_collage(
            styled,
            layout,
            int(spacing),
            int(outer_margin),
            background,
            int(items_per_line),
        )
        output = torch.from_numpy(np.asarray(collage).copy()).to(dtype=torch.float32) / 255.0
        return io.NodeOutput(output.unsqueeze(0))


__all__ = [
    "IMAGE_INPUT_NAMES",
    "LabeledImageCollage",
    "compose_collage",
    "draw_label",
    "font_options",
    "font_records",
    "label_extension_size",
    "load_font",
    "prepare_images",
]
