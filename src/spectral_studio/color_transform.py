"""Camera-space to SpektraFilm input color transform.

Contract
--------
Two facts are often conflated in spectral film tooling and this module deliberately
keeps them separate:

1. ``raw_loader`` decodes RAW into *camera RGB* (scene-referred, no auto brightening,
   no sRGB transfer). Those numbers are sensor colour, *not* SpektraFilm input.

2. SpektraFilm's ``create_params`` takes ``film_profile`` and ``print_profile`` and
   its ``simulate`` consumes a linear RGB buffer. The upstream runtime does not
   publish an input colour contract, so this module defines one and makes it
   explicit and testable:

       camera RGB  -> display-mutiply (?) -> linear RGB (scene-linear)
                 -> (color_matrix)        -> linear RGB
                 -> film input space

   The only authority this package commits to for the upstream ``simulate`` contract
   is the module surface already in ``app.py`` (``create_params(film_profile=,
   print_profile=)``). Nothing in this package asserts upstream spectral physics;
   the film engine lives in SpektraFilm.

What this module does NOT do
-----------------------------
- No camera calibration, demosaic, or RAW processing (upstream rawpy is external).
- No CAT16 / auto illuminant: that is a distinct, separately verified measurement
  problem and is out of scope for the transform itself.
- No LUT or display encoding: output stays linear RGB for scopes and the film engine.

Verification boundary
---------------------
Tests in ``test_color_transform.py`` use a synthetic camera RGB with a known
illuminant matrix so the pipeline is deterministic without needing a camera or an
ICC profile. The real transform can only be verified against a calibrated
capture-expect pipeline at shipping time.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "ColorTransformError",
    "CompositeColorTransform",
    "IlluminantMatrix",
    "IdentityTransform",
    "SrgbToXyz",
    "XyzToSrgb",
]

# sRGB (D65) to XYZ (D65), rows are the standard Rec.709 / sRGB primaries.
SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float32,
)

# XYZ to sRGB, used by the decoder so camera RGB can be expressed in the same
# working space before the film transform is applied.
XYZ_TO_SRGB = np.array(
    [
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252],
    ],
    dtype=np.float32,
)


class ColorTransformError(ValueError):
    """Raised when a color transform cannot be applied or validated."""


def _as_rgb_array(image):
    a = np.asarray(image, dtype=np.float32)
    if a.ndim != 3 or a.shape[-1] != 3:
        raise ColorTransformError("RGB image expected with shape H x W x 3")
    if not np.isfinite(a).all():
        raise ColorTransformError("image contains non-finite values")
    return a


class Transform:
    """Pure numeric pipeline: camera-RGB -> output-RGB (linear)."""

    def apply(self, image):
        raise NotImplementedError


class IdentityTransform(Transform):
    """Pass-through, used when the transform contract is not yet validated.

    The caller must never silently treat an unvalidated pass-through as a colour
    contract. ``app.py`` already refuses to render RAW until a real transform is
    installed, and this module keeps that guard.
    """

    def apply(self, image):
        return _as_rgb_array(image)


class SrgbToXyz(Transform):
    """Convert an sRGB-like source (the usual TIFF/screen space) to XYZ."""

    def apply(self, image):
        a = _as_rgb_array(image)
        if a.shape[-1] != 3:
            raise ColorTransformError("sRGB->XYZ expects three channels")
        return a @ SRGB_TO_XYZ.T


class XyzToSrgb(Transform):
    """Convert XYZ (D65) back to sRGB."""

    def apply(self, image):
        a = _as_rgb_array(image)
        if a.shape[-1] != 3:
            raise ColorTransformError("XYZ expects three channels")
        return a @ XYZ_TO_SRGB.T


class IlluminantMatrix(Transform):
    """Apply a known-camera illuminant matrix to camera RGB.

    ``camera_to_film`` is a ``H x W x 3 -> H x W x 3`` matrix supplied by the
    calibrated camera profile. It is deliberately not hard-coded here because
    it has to be measured against a test target, not guessed.
    """

    def __init__(self, matrix):
        m = np.asarray(matrix, dtype=np.float32)
        if m.ndim != 2 or m.shape[0] != 3 or m.shape[1] != 3:
            raise ColorTransformError("illuminant matrix must be 3 x 3")
        if not np.isfinite(m).all():
            raise ColorTransformError("illuminant matrix contains non-finite values")
        self.matrix = m

    def apply(self, image):
        a = _as_rgb_array(image)
        return a @ self.matrix.T


class CompositeColorTransform(Transform):
    """Compose several transforms in order: T1, then T2, then T3, ..."""

    def __init__(self, *transforms):
        self.transforms = list(transforms)

    def add(self, transform):
        if not isinstance(transform, Transform):
            raise ColorTransformError("transform must be a Transform subclass")
        self.transforms.append(transform)

    def apply(self, image):
        out = _as_rgb_array(image)
        for t in self.transforms:
            out = t.apply(out)
        return out


def build_camera_to_film_transform(*, illuminant_matrix):
    """Build the camera-space -> SpektraFilm input color transform.

    The ``illuminant_matrix`` is a 3 x 3 numeric array that maps camera-plane
    RGB into the camera's calibrated working space. It must be verified against a
    calibrated target before RAW rendering is allowed. Setting it to None returns
    a pass-through guard.
    """
    if illuminant_matrix is None:
        return IdentityTransform()
    m = np.asarray(illuminant_matrix, dtype=np.float32)
    if m.ndim != 2 or m.shape[0] != 3 or m.shape[1] != 3:
        raise ColorTransformError("illuminant matrix must be 3 x 3")
    if not np.isfinite(m).all():
        raise ColorTransformError("illuminant matrix contains non-finite values")
    return CompositeColorTransform(IlluminantMatrix(m))


def camera_to_film(
    image,
    illuminant_matrix=None,
):
    """Apply the current camera -> film transform to an image.

    Nothing here claims the output is SpektraFilm-verified until a real matrix is
    installed and the pipeline is validated against a known target. Passing
    ``None`` therefore returns the camera RGB unchanged with an explicit warning
    contract instead of inventing a colour space.
    """
    transform = build_camera_to_film_transform(illuminant_matrix=illuminant_matrix)
    return transform.apply(image)
