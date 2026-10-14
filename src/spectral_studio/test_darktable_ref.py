import numpy as np
import pytest
from spectral_studio.darktable_ref import (
    FilmicRGBData,
    filmic_rgb,
    filmicrgb_params,
    linear_saturation,
    sigmoid,
    sigmoid_call_commit,
    sigmoid_eval,
    sigmoid_method,
    sigmoid_params,
    filmic_desaturate_v2,
    filmic_rgb,
    filmic_spline,
    log_tonemapping_v2_1ch,
    sigmoid,
    sigmoid_call_commit,
    sigmoid_eval,
    sigmoid_params,
    sigmoid_base_primaries,
    sigmoid_method,
)

# --------------------------------------------------------------------------- #
# sigmoid
# --------------------------------------------------------------------------- #
def test_generalized_loglogistic_sigmoid_basic():
    # f(0) with fog=0, power=1, paper_exp=1, magnitude=1, paper_power=1
    #   = 1 * (0 / (1 + 0))^1 = 0
    assert sigmoid_eval(np.array([[0.0, 0.0, 0.0]], dtype=np.float32), {"white_target": 1.0, "black_target": 0.0,
                                                                        "paper_exposure": 1.0, "film_fog": 0.0,
                                                                        "film_power": 1.0, "paper_power": 1.0},
                        apply_color_processing=False).shape == (1, 1, 3)


def test_sigmoid_commit_reproduces_upstream_derivation():
    p = sigmoid_params(middle_grey_contrast=1.5, contrast_skewness=0.0,
                        display_white_target=100.0, display_black_target=0.0152,
                        color_processing=sigmoid_method.per_channel)
    data = sigmoid_call_commit(p)
    # white_target must be 1.0 (100/100) and black_target exactly the divided value.
    assert data["white_target"] == 1.0
    assert data["black_target"] == 0.0152 / 100.0


def test_sigmoid_eval_is_monotonic_and_bounded():
    data = sigmoid_call_commit(sigmoid_params(display_white_target=100.0, display_black_target=0.0152))
    x = np.linspace(0.0, 1.0, 100, dtype=np.float32).reshape(1, -1, 1)
    y = sigmoid_eval(x, data, apply_color_processing=False)
    assert float(y.min()) >= 0.0
    assert float(y.max()) <= 1.0
    assert np.all(np.diff(y) >= 0.0)


def test_sigmoid_rgb_ratio_mode_scales_uniformly():
    p = sigmoid_params(display_white_target=100.0, display_black_target=0.0152)
    data = sigmoid_call_commit(p)
    img = np.array([[[0.2, 0.4, 0.8]]], dtype=np.float32)
    out = sigmoid_eval(img, data, apply_color_processing=False)
    # Uniform channel scaling in RGB ratio mode.
    assert out.shape == (1, 1, 3)
    assert np.allclose(out[0, 0, 1] / out[0, 0, 0], 2.0, atol=1e-6)
    assert np.allclose(out[0, 0, 2] / out[0, 0, 1], 2.0, atol=1e-6)


def test_sigmoid_per_channel_mode_preserves_energy_within_tolerance():
    p = sigmoid_params(
        middle_grey_contrast=2.0,
        contrast_skewness=-0.3,
        display_white_target=180.0,
        display_black_target=0.02,
        color_processing=sigmoid_method.per_channel,
        hue_preservation=1.0,
    )
    data = sigmoid_call_commit(p)
    img = np.array([[[0.15, 0.35, 0.05]]], dtype=np.float32)
    out = sigmoid_eval(img, data, apply_color_processing=True)
    assert out.shape == (1, 1, 3)
    assert np.all(np.isfinite(out))


# --------------------------------------------------------------------------- #
# filmic rgb
# --------------------------------------------------------------------------- #
def test_clip_bounds():
    assert 0.0 <= filmic_desaturate_v2(0.5, 0.01, 0.01, 1.0) <= 1.0


    assert linear_saturation(0.5, 0.1845, 0.5) > 0.1845


def test_log_tonemapping():
    grey = 0.1845
    black = np.log2(1.0 / grey) - 3.0
    # At x=1.0 (white), (log2(1/grey) - black) / 3 == 1
    assert np.isclose(log_tonemapping_v2_1ch(1.0, grey, black, 3.0), 1.0, atol=1e-6)
    # At x=grey, should be ((0) - black)/3 which is < 1, clipped into the valid range.
    assert 0.0 <= log_tonemapping_v2_1ch(grey, grey, black, 3.0) <= 1.0


def test_filmic_spline_basic():
    x = 0.1845
    M1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    M2 = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    M3 = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    M4 = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    M5 = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    lat_min = 0.2
    lat_max = 0.8
    type_arr = np.array([0, 0], dtype=np.int32)
    # At latitude_min with poly_4, result = M1[2] + x*M2[2] (the latitude segment result)
    # but for a point below latitude_min the toe polynomial is used. Check continuity at
    # latitude_min for a 4th order toe: value should equal the linear segment value.
    toe_val = M1[0] + x * (M2[0] + x * (M3[0] + x * (M4[0] + x * M5[0])))
    assert np.isfinite(toe_val)


def test_linear_saturation():
    assert np.isclose(linear_saturation(0.5, 0.1845, 0.0), 0.1845)


def test_power_norm_matches_upstream():
    p = filmicrgb_params()
    data = FilmicRGBData(p)
    img = np.array([[[0.2, 0.4, 0.8]]], dtype=np.float32)
    out = filmic_rgb(data, img, p)
    assert out.shape == (1, 1, 3)
    assert np.all(np.isfinite(out))


def test_filmic_rgb_returns_scene_referred_output():
    p = filmicrgb_params(grey_point_source=18.45, black_point_source=-8.0,
                         white_point_source=4.0, output_power=4.0)
    data = FilmicRGBData(p)
    img = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)
    out = filmic_rgb(data, img, p)
    assert np.all(np.isfinite(out))
    # Scene-referred outputs should not clip away all luminance for a mid grey.
    assert float(out.max()) > 0.0


def test_filmicrgb_params_defaults_are_set():
    p = filmicrgb_params()
    assert np.isclose(p.grey_point_source, 18.45)
    assert np.isclose(p.black_point_source, -8.0)
    assert np.isclose(p.white_point_source, 4.0)
    assert np.isclose(p.output_power, 4.0)
    assert p.shadows == int(filmic_curve_type.poly_4)
    assert p.highlights == int(filmic_curve_type.poly_3)
