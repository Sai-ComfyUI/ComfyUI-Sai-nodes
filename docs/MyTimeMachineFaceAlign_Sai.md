# YuNet FFHQ Face Align Ψ

Detects the selected face with OpenCV YuNet and produces an FFHQ-style aligned
1024×1024 crop suitable for MyTimeMachine.

## Inputs

- `image`: original full image or portrait batch.
- `face_index`: faces are sorted largest-first; zero selects the largest face.
- `crop_scale`: higher values retain more surroundings and make the head smaller.
- `detector_confidence`: lower this only when a valid face is not detected.

## Outputs

- `aligned_face`: connect to any algorithm that expects an FFHQ-aligned face,
  including **MyTM Re-ageing Ψ**.
- `reframe_context`: connect to **YuNet FFHQ Face Restore Ψ**.

The detector model is `models/mytimemachine/face_detection_yunet_2026may.onnx`.
