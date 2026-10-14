import numpy as np
import pytest
from spectral_studio.color_transform import (
    camera_to_film,
    CompositeColorTransform,
    IlluminantMatrix,
    SrgbToXyz,
    XyzToSrgb,
)


def test_identity_passes_through_empty_rgb():
    assert camera_to_film(np.zeros((4, 4, 3), dtype=np.float32)).shape == (4, 4, 3)


def test_illuminant_matrix_applies_known_values():
    # A = 2*identity: every camera RGB channel should exactly double.
    m = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]], dtype=np.float32)
    img = np.array([[[1.0, 0.0, 0.5]]], dtype=np.float32)
    out = camera_to_film(img, illuminant_matrix=m)
    expected = np.array([[[2.0, 0.0, 1.0]]], dtype=np.float32)
    assert np.array_equal(out, expected)


def test_composite_srgb_to_xyz_and_back_is_close():
    srgb = np.array([[[0.5, 0.2, 0.9]]], dtype=np.float32)
    back = XyzToSrgb().apply(SrgbToXyz().apply(srgb))
    assert np.allclose(back, srgb, atol=1e-6)


def test_camera_to_film_rejects_non_rgb():
    from spectral_studio.color_transform import ColorTransformError
    with pytest.raises(ColorTransformError):
        camera_to_film(np.zeros((4, 4), dtype=np.float32))


def test_camera_to_film_rejects_nonfinite():
    from spectral_studio.color_transform import ColorTransformError
    img = np.zeros((4, 4, 3), dtype=np.float32)
    img[0, 0, 0] = np.nan
    with pytest.raises(ColorTransformError):
        camera_to_film(img)
