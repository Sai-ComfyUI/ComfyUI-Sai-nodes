# YuNet FFHQ Face Restore Ψ

Inverse-warps the generated aligned face back into the original image.

## Inputs

- `processed_face`: processed output corresponding to the aligned face.
- `reframe_context`: context from **YuNet FFHQ Face Align Ψ**.
- `mask_shape`: oval avoids visible square corners; full crop retains more hair
  and neck but may expose a rectangular seam.
- `mask_scale`: controls how much of the generated crop is pasted.
- `feather`: width of the soft transition.
- `color_match_strength`: matches generated crop statistics to the original.

## Outputs

- `composited_image`: original-resolution image with the transformed face.
- `paste_mask`: the final source-resolution blend mask.
