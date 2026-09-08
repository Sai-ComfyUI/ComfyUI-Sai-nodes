# Labeled Image Collage Ψ

Creates one horizontal or vertical collage with a shared label style.

## Quick use

1. Connect images to `image_1`, `image_2`, and the growing sockets that follow.
2. Enter one label per line in `labels`. IMAGE batches are flattened in socket order.
3. Choose `uniform cells (fit)` for equal cells with background padding, or `match cross-axis` to preserve aspect ratio while matching height (horizontal) or width (vertical).
4. Set `items_per_line` to wrap horizontal layouts into additional rows or vertical layouts into additional columns.
5. Set the shared label placement, position, direction, font, colors, background extent, padding, and outline.

Blank or missing label lines leave the corresponding image unlabeled.

## Grid wrapping

`items_per_line` is the maximum number of columns in a horizontal layout, or the maximum number of rows in a vertical layout. Overflow continues on the next row or column. The final incomplete line is kept as a rectangular grid, and its empty cells use `background_color`.

## Named presets

Every node has `preset`, `save preset`, and `delete preset` controls. A named preset stores the multiline `labels` content together with the layout, sizing, spacing, colors, label placement, typography, and font settings. It excludes IMAGE connections.

Presets are stored in browser localStorage for the current ComfyUI address. They are available across workflows in the same browser profile, but are not embedded in workflow JSON or automatically synchronized to other browsers or computers.

## Label placement

`label_placement` determines whether labels cover the source image:

- `overlay`: draws the label over the image. `label_background_extent` controls whether its background is a text box or a full-axis banner.
- `extend top` / `extend bottom`: adds a horizontal label strip outside the image.
- `extend left` / `extend right`: adds a vertical label strip outside the image.

Extend modes preserve every original image pixel. The node calculates one shared strip thickness from the largest label, then applies it to every item so collage cells remain aligned. `label_background_color` fills the entire added strip; `label_background_extent` only affects overlay mode. For top/bottom strips, the horizontal part of `label_position` controls text alignment. For left/right strips, its vertical part controls alignment.

## Chinese and other CJK text

The `font_name` menu lists discovered font files and readable face names, including individual TTC/OTC faces. `Auto (CJK preferred)` prioritizes a Traditional Chinese sans-serif face. The node searches `assets/fonts`, `incoming`, the legacy `temp` exchange folder, and common operating-system CJK fonts. Use `font_path` only for an unlisted TTF, OTF, TTC, or OTC file; the most suitable Traditional Chinese face is selected automatically.

## Color alpha and banners

The alpha channel selected by the canvas-background, text, label-background, and outline color pickers is applied directly. There is no separate label-background opacity control.

When `background_color` is translucent, the node outputs an RGBA IMAGE: fitted-cell padding, spacing, and outer margins retain the selected alpha while source image pixels remain opaque. With an opaque canvas background and RGB sources, the output remains RGB.

`label_background_extent` controls the background shape:

- `text box`: background follows the text and padding.
- `full width banner`: background fills the image width while its vertical position follows `label_position`.
- `full height banner`: background fills the image height while its horizontal position follows `label_position`.

For future user/agent file exchange, place source files under the Git-ignored `incoming/` directory. Accepted assets should then be moved into their permanent project location.
