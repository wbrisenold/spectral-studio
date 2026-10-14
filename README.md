# Spectral Studio

A Qt desktop studio for spectral film simulation and image processing, backed by the original SpektraFilm runtime (`create_params` / `simulate`), with a SQLite non-destructive culler, exposure helpers, and on-image scopes.

> **Status: do not ship.** This repository is an audited snapshot with an incomplete UI and several verified blockers. See `STATUS.md` for the full list of open items.

## What it can do today

- Load verified, connected controls: raw camera RGB, paper/film selector, EV, generic exposure suggestion.
- Render film output via upstream SpektraFilm (open-source, GPLv3) — spectral simulation and hardcoded upstream profiles only.
- Non-destructive library: import a folder, rate 0–5 stars, pick/reject with SQLite persistence.
- Scopes: histogram, waveform, RGB parade, vectorscope, clipping measurements.
- Export scene-linear float TIFF.

## Windows / Linux / macOS

Run from source after building the environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spectral_studio.desktop
```

`requirements.txt` is generated from `pyproject.toml` for local development. The production entry point is:

```bash
pip install .
spectral-studio
```

## How to run tests

```bash
PYTHONPATH=src python -m pytest -q src/spectral_studio/test_scene_tools.py
```

## License

See `LICENSE` for details. Original darktable sources are GPLv3-or-later; preserve upstream notices and verify the license before redistribution.

## Lawyering in (required before shipping)

- Implement and verify a camera-space to SpektraFilm input-space conversion before enabling RAW rendering.
- Provide the original darktable `sigmoid`, `filmicrgb`, and `AgX` implementations if display color encoding parity is required.
- Validate against physical film or ISO test targets before releasing film reproductions.
- Keep the public wrapper as an original-code package; upstream spectral physics lives in SpektraFilm, not here.
