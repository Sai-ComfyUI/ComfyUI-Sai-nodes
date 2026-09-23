"""Streaming LumaFlux video conversion and HDR10 delivery nodes."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import io as stdlib_io
import os
from pathlib import Path
import subprocess
from typing import Callable, Iterator

import av
import comfy.model_management as model_management
import comfy.utils
import folder_paths
import numpy as np
import torch
from comfy_api.latest import io, ui

from .lumaflux import (
    RESIZE_MODES,
    LumaFluxModelHandle,
    LumaFluxModelType,
    _save_exr_float32,
    _save_png16_pq,
    _write_hdr_sidecar,
    convert_sdr_to_hdr,
    normalize_boolean_input,
    resolve_lumaflux_geometry,
)


HDRVideoStreamType = io.Custom("HDR_VIDEO_STREAM")
_AUDIO_AAC = "Preserve source audio (AAC)"
_AUDIO_NONE = "No audio"
HDR_VIDEO_FORMAT = "HDR10 MP4 (HEVC Main10)"
HDR_EXR_SEQUENCE_FORMAT = "OpenEXR sequence (32-bit float, linear BT.2020)"
HDR_PNG_SEQUENCE_FORMAT = "PNG sequence (16-bit, PQ BT.2020)"


@dataclass(frozen=True)
class HDRVideoStream:
    """Deferred, reusable contract for an HDR frame stream.

    Frames are produced one at a time as BHWC PQ/BT.2020 tensors. The contract
    is intentionally independent of LumaFlux so future SDR-to-HDR engines can
    feed the same saver.
    """

    frame_iterator_factory: Callable[[], Iterator[torch.Tensor]]
    frame_rate: Fraction
    frame_count: int
    source_size: tuple[int, int]
    processed_size: tuple[int, int]
    source_video: object
    transfer: str = "pq"
    primaries: str = "bt2020"
    peak_nits: float = 1000.0
    geometry_notice: str = ""

    def iter_frames(self) -> Iterator[torch.Tensor]:
        return self.frame_iterator_factory()


def _frame_time(frame: av.VideoFrame, stream: av.VideoStream, fallback_index: int, fps: float) -> float:
    if frame.pts is not None and stream.time_base is not None:
        return float(frame.pts * stream.time_base)
    return fallback_index / fps


def iter_video_frames(video, expected_size: tuple[int, int]) -> Iterator[torch.Tensor]:
    """Decode a native ComfyUI VIDEO source without materializing an IMAGE batch."""

    source = video.get_stream_source()
    if isinstance(source, stdlib_io.BytesIO):
        source.seek(0)
    start_time, duration = video.get_active_trim_window()
    end_time = start_time + duration if duration > 0 else None

    with av.open(source, mode="r") as container:
        if not container.streams.video:
            raise ValueError("VIDEO does not contain a decodable video stream.")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        fps = float(stream.average_rate or video.get_frame_rate() or 1)
        if start_time > 0 and stream.time_base is not None:
            container.seek(max(0, int(start_time / stream.time_base)), backward=True, stream=stream)

        emitted = 0
        for decoded_index, frame in enumerate(container.decode(stream)):
            timestamp = _frame_time(frame, stream, decoded_index, fps)
            if timestamp + (0.5 / fps) < start_time:
                continue
            if end_time is not None and timestamp >= end_time:
                break
            array = frame.to_ndarray(format="rgb24")
            actual_size = (int(array.shape[1]), int(array.shape[0]))
            if actual_size != expected_size:
                raise ValueError(
                    "The streamed VIDEO dimensions differ from its declared dimensions "
                    f"({actual_size[0]}x{actual_size[1]} vs {expected_size[0]}x{expected_size[1]}). "
                    "Connect Load Video or Trim Video directly; cropped/rotated wrappers are not yet supported."
                )
            emitted += 1
            yield torch.from_numpy(np.ascontiguousarray(array)).to(torch.float32).div_(255.0).unsqueeze(0)
        if emitted == 0:
            raise ValueError("VIDEO produced no frames inside its active trim window.")


def _master_display(peak_nits: float) -> str:
    maximum = int(round(peak_nits * 10_000.0))
    max_fall = min(int(round(peak_nits)), 400)
    return (
        "master-display=G(8500,39850)B(6550,2300)R(35400,14600)"
        f"WP(15635,16450)L({maximum},1):max-cll={int(round(peak_nits))},{max_fall}"
    )


def _validate_x265() -> None:
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-h", "encoder=libx265"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to save HDR10 video.") from exc
    if result.returncode != 0 or "yuv420p10le" not in result.stdout:
        raise RuntimeError("The ffmpeg on PATH must include a 10-bit-capable libx265 encoder.")


class HDR10VideoWriter:
    """Write incoming PQ RGB frames directly to a 10-bit HEVC ffmpeg pipe."""

    def __init__(
        self,
        path: Path,
        width: int,
        height: int,
        frame_rate: Fraction,
        crf: int,
        preset: str,
        peak_nits: float,
        source_video,
        preserve_audio: bool,
    ) -> None:
        _validate_x265()
        if width % 2 or height % 2:
            raise ValueError(
                f"HDR10 yuv420p10le requires even dimensions; received {width}x{height}. "
                "Use resize mode, or export an EXR/PNG sequence to preserve odd source dimensions."
            )
        self.path = path
        self.width = width
        self.height = height
        self._closed = False
        source = source_video.get_stream_source()
        if preserve_audio and not isinstance(source, str):
            raise ValueError("Audio preservation requires a disk-backed VIDEO from Load Video.")

        command = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb48le",
            "-s", f"{width}x{height}", "-r", str(frame_rate), "-i", "pipe:0",
        ]
        if preserve_audio:
            start_time, duration = source_video.get_active_trim_window()
            if start_time > 0:
                command.extend(["-ss", f"{start_time:.9f}"])
            if duration > 0:
                command.extend(["-t", f"{duration:.9f}"])
            command.extend(["-i", source, "-map", "0:v:0", "-map", "1:a?"])
        command.extend([
            "-c:v", "libx265", "-crf", str(crf), "-preset", preset,
            "-pix_fmt", "yuv420p10le",
            "-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc",
            "-x265-params",
            "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:" + _master_display(peak_nits),
            "-tag:v", "hvc1",
        ])
        if preserve_audio:
            command.extend(["-c:a", "aac", "-b:a", "192k", "-shortest"])
        else:
            command.append("-an")
        command.extend(["-movflags", "+faststart", str(path)])
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: torch.Tensor) -> None:
        if self._closed or self._process.stdin is None:
            raise RuntimeError("Cannot write to a closed HDR video writer.")
        if frame.ndim != 4 or tuple(frame.shape) != (1, self.height, self.width, 3):
            raise ValueError(f"Expected one BHWC frame at {self.width}x{self.height}; received {tuple(frame.shape)}.")
        array = (
            frame[0].detach().cpu().float().clamp(0.0, 1.0).mul(65535.0).round().numpy().astype("<u2")
        )
        try:
            self._process.stdin.write(np.ascontiguousarray(array).tobytes())
        except BrokenPipeError as exc:
            error = self._process.stderr.read().decode("utf-8", errors="replace") if self._process.stderr else ""
            raise RuntimeError(f"ffmpeg stopped while encoding HDR video: {error.strip()}") from exc

    def close(self) -> Path:
        if self._closed:
            return self.path
        self._closed = True
        if self._process.stdin is not None:
            self._process.stdin.close()
        return_code = self._process.wait()
        error = self._process.stderr.read().decode("utf-8", errors="replace") if self._process.stderr else ""
        if self._process.stderr is not None:
            self._process.stderr.close()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}: {error.strip()}")
        return self.path

    def abort(self) -> None:
        if not self._closed:
            self._closed = True
            if self._process.stdin is not None:
                self._process.stdin.close()
            self._process.terminate()
            self._process.wait()
            if self._process.stderr is not None:
                self._process.stderr.close()
        self.path.unlink(missing_ok=True)


class LumaFluxSdrVideoToHdr(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LumaFluxSdrVideoToHdr_Sai",
            display_name="LumaFlux SDR Video to HDR Ψ",
            category="Sai/SDR to HDR/LumaFlux",
            description=(
                "Creates a deferred HDR video stream from ComfyUI VIDEO. Frames are processed sequentially with "
                "shared bridge noise and previous-frame RQS spline EMA; no IMAGE batch is created."
            ),
            inputs=[
                LumaFluxModelType.Input("lumaflux_model"),
                io.Video.Input("video"),
                io.Combo.Input(
                    "resize_mode",
                    options=RESIZE_MODES,
                    default="crop",
                    tooltip=(
                        "none: require exact ×16; crop: edge-pad for inference then restore the original size; "
                        "resize: resample to the nearest ×16 size."
                    ),
                ),
                io.Int.Input("steps", default=8, min=1, max=50),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.Boolean.Input(
                    "shared_noise",
                    default=True,
                    tooltip=(
                        "Reuse bridge noise and carry the previous applied RQS spline through EMA. "
                        "Disable to process every frame independently."
                    ),
                ),
                io.Float.Input("spline_ema", default=0.8, min=0.0, max=0.99, step=0.01),
                io.Float.Input("time_shift_factor", default=1.0, min=0.01, max=10.0, step=0.01, advanced=True),
                io.Float.Input("bridge_noise", default=0.05, min=0.0, max=1.0, step=0.001, advanced=True),
                io.Float.Input("tone_strength", default=1.0, min=0.0, max=2.0, step=0.01, advanced=True),
            ],
            outputs=[HDRVideoStreamType.Output("hdr_video_stream")],
        )

    @classmethod
    def execute(
        cls, lumaflux_model, video, resize_mode, steps, seed, shared_noise, spline_ema,
        time_shift_factor, bridge_noise, tone_strength,
    ) -> io.NodeOutput:
        if not isinstance(lumaflux_model, LumaFluxModelHandle):
            raise TypeError("lumaflux_model must come from Apply LumaFlux Adapter Ψ.")
        if not 0.0 <= spline_ema < 1.0:
            raise ValueError("spline_ema must be in [0, 1).")
        source_size = tuple(int(value) for value in video.get_dimensions())
        _model_size, processed_size, geometry_notice = resolve_lumaflux_geometry(
            source_size[0], source_size[1], resize_mode
        )
        frame_rate = Fraction(video.get_frame_rate())
        try:
            frame_count = int(video.get_frame_count())
        except (ValueError, av.error.FFmpegError):
            frame_count = max(1, int(round(float(video.get_duration()) * float(frame_rate))))
        use_shared_noise = normalize_boolean_input(shared_noise, "shared_noise")

        def frame_iterator() -> Iterator[torch.Tensor]:
            shared_bridge_noise = None
            previous_params = None
            progress = comfy.utils.ProgressBar(max(1, frame_count * steps))
            for frame_index, image in enumerate(iter_video_frames(video, source_size)):
                model_management.throw_exception_if_processing_interrupted()
                result = convert_sdr_to_hdr(
                    lumaflux_model,
                    image,
                    steps,
                    seed if use_shared_noise else (seed + frame_index) & 0xFFFFFFFFFFFFFFFF,
                    time_shift_factor,
                    bridge_noise,
                    tone_strength,
                    resize_mode=resize_mode,
                    noise=shared_bridge_noise if use_shared_noise else None,
                    previous_spline_params=previous_params if use_shared_noise else None,
                    spline_ema=spline_ema if use_shared_noise else 0.0,
                    return_continuity_state=True,
                    progress_bar=progress,
                )
                if use_shared_noise and shared_bridge_noise is None:
                    shared_bridge_noise = result["bridge_noise"]
                previous_params = result["spline_params"] if use_shared_noise else None
                yield result["images"]

        temporal_notice = (
            f"Temporal consistency on: shared noise + applied RQS EMA {spline_ema:.2f}."
            if use_shared_noise
            else "Temporal consistency off: frames are independent."
        )
        notice = (
            f"Deferred stream: {frame_count} frames at {frame_rate} fps; "
            f"output {processed_size[0]}x{processed_size[1]}.\n{geometry_notice}\n{temporal_notice}"
        )
        stream = HDRVideoStream(
            frame_iterator, frame_rate, frame_count, source_size, processed_size, video,
            peak_nits=lumaflux_model.peak_nits,
            geometry_notice=geometry_notice,
        )
        return io.NodeOutput(stream, ui=ui.PreviewText(notice))


class SaveHdr10Video(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SaveHdr10Video_Sai",
            display_name="Save HDR10 Video Ψ",
            category="Sai/Saver/HDR",
            description=(
                "Streams HDR_VIDEO_STREAM to 10-bit HEVC HDR10 with mastering metadata, "
                "or to lossless EXR / 16-bit PQ PNG image sequences."
            ),
            is_output_node=True,
            inputs=[
                HDRVideoStreamType.Input("hdr_video_stream"),
                io.String.Input("filename_prefix", default="HDR/LumaFlux_video"),
                io.Combo.Input(
                    "format",
                    options=[HDR_VIDEO_FORMAT, HDR_EXR_SEQUENCE_FORMAT, HDR_PNG_SEQUENCE_FORMAT],
                    default=HDR_VIDEO_FORMAT,
                ),
                io.Int.Input("crf", default=16, min=0, max=51),
                io.Combo.Input("preset", options=["slow", "medium", "fast"], default="medium"),
                io.Combo.Input("audio", options=[_AUDIO_AAC, _AUDIO_NONE], default=_AUDIO_AAC),
            ],
            outputs=[],
        )

    @classmethod
    def execute(
        cls,
        hdr_video_stream,
        filename_prefix: str,
        format: str,
        crf: int,
        preset: str,
        audio: str,
    ) -> io.NodeOutput:
        if not isinstance(hdr_video_stream, HDRVideoStream):
            raise TypeError("hdr_video_stream must be an HDR_VIDEO_STREAM value.")
        if hdr_video_stream.transfer != "pq" or hdr_video_stream.primaries != "bt2020":
            raise ValueError("HDR_VIDEO_STREAM must contain PQ/BT.2020 frames.")
        output_dir = folder_paths.get_output_directory()
        folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, output_dir, *hdr_video_stream.processed_size,
        )
        if format == HDR_VIDEO_FORMAT:
            output_name = f"{filename}_{counter:05}.mp4"
            output_path = Path(folder) / output_name
            writer = HDR10VideoWriter(
                output_path,
                *hdr_video_stream.processed_size,
                hdr_video_stream.frame_rate,
                crf,
                preset,
                hdr_video_stream.peak_nits,
                hdr_video_stream.source_video,
                preserve_audio=audio == _AUDIO_AAC,
            )
            frames_written = 0
            try:
                for frame in hdr_video_stream.iter_frames():
                    writer.write(frame)
                    frames_written += 1
                writer.close()
            except BaseException:
                writer.abort()
                raise
            relative = os.path.join(subfolder, output_name) if subfolder else output_name
            message = (
                f"Saved HDR10 master: {relative}\n"
                f"{frames_written} frames, {hdr_video_stream.processed_size[0]}x{hdr_video_stream.processed_size[1]}, "
                f"{hdr_video_stream.frame_rate} fps, HEVC Main10, PQ/BT.2020."
            )
            return io.NodeOutput(ui=ui.PreviewText(message))

        if format not in (HDR_EXR_SEQUENCE_FORMAT, HDR_PNG_SEQUENCE_FORMAT):
            raise ValueError(f"Unsupported HDR stream format: {format}")
        sequence_name = f"{filename}_{counter:05}"
        sequence_folder = Path(folder) / sequence_name
        sequence_folder.mkdir(parents=True, exist_ok=False)
        frames_written = 0
        extension = "exr" if format == HDR_EXR_SEQUENCE_FORMAT else "png"
        payload = {
            "peak_nits": hdr_video_stream.peak_nits,
            "mastering_display": "G(0.1700,0.7970)B(0.1310,0.0460)R(0.7080,0.2920)WP(0.3127,0.3290)",
        }
        try:
            for frame_index, frame in enumerate(hdr_video_stream.iter_frames()):
                model_management.throw_exception_if_processing_interrupted()
                output_path = sequence_folder / f"{sequence_name}_{frame_index:06d}.{extension}"
                image = frame[0]
                if format == HDR_EXR_SEQUENCE_FORMAT:
                    _save_exr_float32(image, str(output_path), hdr_video_stream.peak_nits)
                    encoding = (
                        f"linear BT.2020 RGB; 1.0 = {hdr_video_stream.peak_nits:g} cd/m2; float32"
                    )
                else:
                    _save_png16_pq(image, str(output_path))
                    encoding = "SMPTE ST 2084 (PQ) BT.2020 RGB; uint16 code values; PNG cICP"
                _write_hdr_sidecar(str(output_path), payload, encoding)
                frames_written += 1
        except BaseException:
            # Keep completed frames for diagnosis or recovery; never silently remove a partial sequence.
            raise
        relative_folder = os.path.join(subfolder, sequence_name) if subfolder else sequence_name
        return io.NodeOutput(
            ui=ui.PreviewText(
                f"Saved HDR image sequence: {relative_folder}\n"
                f"{frames_written} {extension.upper()} frames, "
                f"{hdr_video_stream.processed_size[0]}x{hdr_video_stream.processed_size[1]}, PQ/BT.2020 source."
            )
        )


__all__ = [
    "HDR10VideoWriter",
    "HDR_EXR_SEQUENCE_FORMAT",
    "HDR_PNG_SEQUENCE_FORMAT",
    "HDR_VIDEO_FORMAT",
    "HDRVideoStream",
    "HDRVideoStreamType",
    "LumaFluxSdrVideoToHdr",
    "SaveHdr10Video",
    "iter_video_frames",
]
