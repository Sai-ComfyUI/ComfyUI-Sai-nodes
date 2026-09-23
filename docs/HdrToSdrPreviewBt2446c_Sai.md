# HDR to SDR Preview Ψ

Creates a BT.709 SDR `IMAGE` preview from a PQ/BT.2020 `HDR_IMAGE`. Available
methods are BT.2446 Method C, extended Reinhard with an adjustable BT.2408-style
SDR white point, and a white-point clip. `white_nits` defaults to 203 and is
used by the Reinhard and clip methods only.

This output is intended for preview, comparison, and ordinary image nodes. It
is not the HDR master and must not be tagged or encoded as HDR.
