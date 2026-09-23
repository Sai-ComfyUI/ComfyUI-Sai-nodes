# Save HDR10 Video Ψ

Consumes `HDR_VIDEO_STREAM` one frame at a time. The `format` menu writes an
MP4 HDR10 delivery file, a float32 linear BT.2020 OpenEXR sequence, or a
lossless 16-bit PQ/BT.2020 PNG sequence with `cICP`. No complete frame batch is
retained in memory.

The video track is HEVC Main10 (`yuv420p10le`) with:

- SMPTE ST 2084 (PQ) transfer;
- BT.2020 primaries and BT.2020 non-constant-luminance matrix;
- a 1000-nit mastering display SEI;
- MaxCLL 1000 and MaxFALL 400 content-light metadata;
- `hvc1` sample-entry tag for player compatibility.

`Preserve source audio (AAC)` keeps the source soundtrack and active trim
window while encoding it to AAC 192 kb/s. This option requires a disk-backed
source from ComfyUI's **Load Video**. Choose `No audio` for a memory-backed
VIDEO source.

The node refuses to run when the available ffmpeg/libx265 build cannot encode
10-bit HEVC, so an 8-bit file cannot be mislabeled as HDR.

EXR and PNG sequences are stored in a numbered subfolder. Every frame includes
its matching JSON HDR sidecar. Sequence export also supports odd source sizes;
MP4 `yuv420p10le` requires even width and height and gives an actionable error
instead of silently removing pixels.
