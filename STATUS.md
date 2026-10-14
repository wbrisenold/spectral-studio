# Integration status — verified limitations

This is an audited snapshot of the existing Spectral Studio source, NOT a complete darktable integration.

## Source verification
- SpektraFilm's original runtime exposes `create_params` and `simulate`: https://github.com/andreavolpato/spektrafilm
- darktable's `src/iop/sigmoid.c` contains actual hue preservation and RGB-ratio/per-channel algorithms; it is not implemented here.
- darktable's `src/iop/filmicrgb.c` contains filmic curve and highlight reconstruction algorithms; it is not implemented here.
- darktable's `src/common/iop_order.c` places sigmoid, AgX, filmicrgb and SpektraFilm in its processing order.

## Current blockers
- No camera-space to verified SpektraFilm input-space conversion: RAW rendering deliberately blocked.
- No verified CAT16/auto illuminant or original darktable exposure integration.
- No original Sigmoid, Filmic RGB or AgX implementation.
- No Intel macOS runtime test, compiled dependencies, or actual film golden renders.
- Existing UI is incomplete. Do not claim full editor or output parity.

## How to run tests
`PYTHONPATH=src python -m pytest -q src/spectral_studio/test_scene_tools.py`

## License
Original darktable sources are GPLv3-or-later; inspect and comply with their license before redistribution. SpektraFilm uses GPLv3; preserve upstream notices.
