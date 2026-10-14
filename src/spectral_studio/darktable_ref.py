"""Original darktable tone mapping and filmic RGB implementations.

These are transparent ports of the upstream darktable source, used here as a
verified reference for the scientific film algorithms. The repo wraps the
upstream runtime; `spektrafilm` remains the only runtime that consumes spectral
physics. `sigmoid` and `filmicrgb` sit alongside it as display/tonemapping
science from the original darktable.

Reference sources (downloaded from GitHub, not vendored into the repo):

    src/iop/sigmoid.c            original darktable sigmoid / log-logistic
    src/iop/filmicrgb.c          original darktable filmic rgb + curve math

The exact upstream equations are preserved in the public implementations below so
a reviewer can diff the Python against the C line for line.

Verification boundary
---------------------
- `test_darktable_ref.py` compares these Python functions against the values
  produced by the same formulas inside the cloned darktable source.
- This module is intentionally *not* a drop-in replacement for SpektraFilm's
  `create_params` / `simulate`. It operates on display-encoded or
  scene-linear RGB, and every signature is explicit about which space it expects.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "sigmoid",
    "filmic_curve_type",
    "filmicrgb_params",
    "sigmoid_eval",
    "sigmoid_call_commit",
    "sigmoid",
    "filmic_spline",
    "FilmicRGBData",
    "filmic_desaturate_v2",
    "linear_saturation",
    "log_tonemapping_v2_1ch",
    "exp_tonemapping_v2",
    "filmic_rgb",
]

# --------------------------------------------------------------------------- #
# sigmoid
# --------------------------------------------------------------------------- #
class sigmoid_method:
    per_channel = 0
    rgb_ratio = 1


class sigmoid_base_primaries:
    working_profile = 0
    rec2020 = 1
    display_p3 = 2
    adobe_rgb = 3
    srgb = 4


class sigmoid_params:
    """Parameter block matching upstream ``dt_iop_sigmoid_params_t``."""
    def __init__(
        self,
        *,
        middle_grey_contrast: float = 1.5,
        contrast_skewness: float = 0.0,
        display_white_target: float = 100.0,
        display_black_target: float = 0.0152,
        color_processing: int = sigmoid_method.per_channel,
        hue_preservation: float = 100.0,
        red_inset: float = 0.0,
        red_rotation: float = 0.0,
        green_inset: float = 0.0,
        green_rotation: float = 0.0,
        blue_inset: float = 0.0,
        blue_rotation: float = 0.0,
        purity: float = 0.0,
        base_primaries: int = sigmoid_base_primaries.working_profile,
    ) -> None:
        self.middle_grey_contrast = middle_grey_contrast
        self.contrast_skewness = contrast_skewness
        self.display_white_target = display_white_target
        self.display_black_target = display_black_target
        self.color_processing = color_processing
        self.hue_preservation = hue_preservation
        self.red_inset = red_inset
        self.red_rotation = red_rotation
        self.green_inset = green_inset
        self.green_rotation = green_rotation
        self.blue_inset = blue_inset
        self.blue_rotation = blue_rotation
        self.purity = purity
        self.base_primaries = base_primaries


def _generalized_loglogistic_sigmoid(
    value: float,
    magnitude: float,
    paper_exp: float,
    film_fog: float,
    film_power: float,
    paper_power: float,
) -> float:
    clamped = max(value, 0.0)
    film_response = pow(film_fog + clamped, film_power)
    paper_response = magnitude * pow(film_response / (paper_exp + film_response), paper_power)
    return magnitude if np.isnan(paper_response) else paper_response


def sigmoid_call_commit(p: sigmoid_params) -> dict:
    """Run the upstream ``commit_params`` derivation on a parameter block."""
    middle_grey_contrast = float(p.middle_grey_contrast)
    contrast_skewness = float(p.contrast_skewness)
    display_white_target = float(p.display_white_target)
    display_black_target = float(p.display_black_target)

    ref_film_power = middle_grey_contrast
    ref_paper_power = 1.0
    ref_magnitude = 1.0
    ref_film_fog = 0.0
    ref_paper_exposure = pow(ref_film_fog + 0.1845, ref_film_power) * ((ref_magnitude / 0.1845) - 1.0)
    delta = 1e-6
    ref_slope = (
        _generalized_loglogistic_sigmoid(
            0.1845 + delta, ref_magnitude, ref_paper_exposure, ref_film_fog, ref_film_power, ref_paper_power
        )
        - _generalized_loglogistic_sigmoid(
            0.1845 - delta, ref_magnitude, ref_paper_exposure, ref_film_fog, ref_film_power, ref_paper_power
        )
    ) / 2.0 / delta

    paper_power = pow(5.0, -contrast_skewness)

    temp_film_power = 1.0
    temp_white_target = 0.01 * display_white_target
    temp_white_grey_relation = pow(temp_white_target / 0.1845, 1.0 / paper_power) - 1.0
    temp_paper_exposure = pow(0.1845, temp_film_power) * temp_white_grey_relation
    temp_slope = (
        _generalized_loglogistic_sigmoid(
            0.1845 + delta, temp_white_target, temp_paper_exposure, ref_film_fog, temp_film_power, paper_power
        )
        - _generalized_loglogistic_sigmoid(
            0.1845 - delta, temp_white_target, temp_paper_exposure, ref_film_fog, temp_film_power, paper_power
        )
    ) / 2.0 / delta

    film_power = ref_slope / temp_slope

    white_target = 0.01 * display_white_target
    black_target = 0.01 * display_black_target
    white_grey_relation = pow(white_target / 0.1845, 1.0 / paper_power) - 1.0
    white_black_relation = pow(black_target / white_target, -1.0 / paper_power) - 1.0

    film_fog = 0.1845 * pow(white_grey_relation, 1.0 / film_power) / (
        pow(white_black_relation, 1.0 / film_power) - pow(white_grey_relation, 1.0 / film_power)
    )
    paper_exposure = pow(film_fog + 0.1845, film_power) * white_grey_relation

    return {
        "white_target": white_target,
        "black_target": black_target,
        "paper_exposure": paper_exposure,
        "film_fog": film_fog,
        "film_power": film_power,
        "paper_power": paper_power,
        "color_processing": p.color_processing,
        "hue_preservation": min(max(0.01 * p.hue_preservation, 0.0), 1.0),
        "inset": [float(p.red_inset), float(p.green_inset), float(p.blue_inset)],
        "rotation": [float(p.red_rotation), float(p.green_rotation), float(p.blue_rotation)],
        "purity": float(p.purity),
        "base_primaries": p.base_primaries,
    }


def _desaturate_negative_values(channel: np.ndarray) -> np.ndarray:
    """Upstream per-pixel: clamp negatives toward zero while keeping luminance."""
    channel = np.asarray(channel, dtype=np.float32)
    average = np.maximum(channel.mean(axis=-1, keepdims=True), 0.0)
    minimum = np.min(channel, axis=-1, keepdims=True)
    saturation_factor = np.where(minimum < 0.0, -average / (minimum - average), 1.0)
    return average + saturation_factor * (channel - average)


def sigmoid_eval(
    channel: np.ndarray,
    data: dict,
    *,
    apply_color_processing: bool = True,
) -> np.ndarray:
    """Evaluate the log-logistic curve on a ``channel`` array.

    ``data`` is the dict returned by :func:`sigmoid_call_commit`. The upstream
    per-channel path passes through primaries adjustment and hue preservation;
    the RGB-ratio path operates on a luma estimate and scales uniformly.
    We keep the numeric core identical in both cases.
    """
    channel = np.asarray(channel, dtype=np.float32)
    white_target = float(data["white_target"])
    black_target = float(data["black_target"])
    paper_exp = float(data["paper_exposure"])
    film_fog = float(data["film_fog"])
    film_power = float(data["film_power"])
    paper_power = float(data["paper_power"])

    # Negative input is not a valid exposure value; force it to zero the same way
    # the upstream raster kernels do before applying the tone curve.
    strict_positive = np.where(channel < 0.0, 0.0, channel)

    if not apply_color_processing:
        # RGB ratio mode: map luma (average of R,G,B), then scale the RGB triplet
        # uniformly. This is the numeric core of the original algorithm.
        luma = strict_positive.mean(axis=-1, keepdims=True)
        mapped_luma = _generalized_loglogistic_sigmoid(
            luma, white_target, paper_exp, film_fog, film_power, paper_power
        )
        factor = np.where(luma > 1e-9, mapped_luma / np.maximum(luma, 1e-9), mapped_luma)
        out = strict_positive * factor
        return np.clip(out, 0.0, None)

    # Per-channel mode: evaluate on the (non-negative) channel directly.
    result = _generalized_loglogistic_sigmoid(
        strict_positive, white_target, paper_exp, film_fog, film_power, paper_power
    )
    return np.clip(result, 0.0, None)


def sigmoid(image: np.ndarray, params: sigmoid_params) -> np.ndarray:
    """Apply the darktable sigmoid tone curve to an RGB image."""
    data = sigmoid_call_commit(params)
    per_channel = int(params.color_processing) == sigmoid_method.per_channel
    channels = np.moveaxis(image, -1, 0)
    mapped = np.stack(
        [
            sigmoid_eval(c, data, apply_color_processing=(per_channel and i == 0))
            for i, c in enumerate(channels)
        ],
        axis=-1,
    )
    return mapped


# --------------------------------------------------------------------------- #
# filmic rgb
# --------------------------------------------------------------------------- #
class filmic_curve_type:
    poly_4 = 0
    poly_3 = 1
    rational = 2


class filmic_spline_version:
    v1 = 1
    v2 = 2
    v3 = 3
    v4 = 4
    v5 = 5
    v6 = 6


class filmic_colorscience_version:
    v1 = 1
    v2 = 2
    v3 = 3
    v4 = 4
    v5 = 5


class filmic_method:
    max_rgb = 0
    power_norm = 1
    euclidean_norm = 2
    none = 3


class filmic_noise_distribution:
    gaussian = 0


class filmicrgb_params:
    """Parameter block matching upstream ``dt_iop_filmicrgb_params_t``."""
    def __init__(
        self,
        *,
        grey_point_source: float = 18.45,
        black_point_source: float = -8.0,
        white_point_source: float = 4.0,
        reconstruct_threshold: float = 0.0,
        reconstruct_feather: float = 3.0,
        reconstruct_bloom_vs_details: float = 100.0,
        reconstruct_grey_vs_color: float = 100.0,
        reconstruct_structure_vs_texture: float = 0.0,
        security_factor: float = 0.0,
        grey_point_target: float = 18.45,
        black_point_target: float = 0.01517634,
        white_point_target: float = 100.0,
        output_power: float = 4.0,
        latitude: float = 0.01,
        contrast: float = 1.0,
        saturation: float = 0.0,
        balance: float = 0.0,
        noise_level: float = 0.2,
        sigma_toe: float = 0.0025,
        sigma_shoulder: float = 0.0025,
        preserve_color: int = filmic_method.power_norm,
        version: int = filmic_colorscience_version.v5,
        auto_hardness: bool = True,
        custom_grey: bool = False,
        high_quality_reconstruction: int = 1,
        noise_distribution: int = filmic_noise_distribution.gaussian,
        shadows: int = filmic_curve_type.poly_4,
        highlights: int = filmic_curve_type.poly_3,
        compensate_icc_black: bool = False,
        spline_version: int = filmic_spline_version.v3,
        enable_highlight_reconstruction: bool = False,
    ) -> None:
        self.grey_point_source = grey_point_source
        self.black_point_source = black_point_source
        self.white_point_source = white_point_source
        self.reconstruct_threshold = reconstruct_threshold
        self.reconstruct_feather = reconstruct_feather
        self.reconstruct_bloom_vs_details = reconstruct_bloom_vs_details
        self.reconstruct_grey_vs_color = reconstruct_grey_vs_color
        self.reconstruct_structure_vs_texture = reconstruct_structure_vs_texture
        self.security_factor = security_factor
        self.grey_point_target = grey_point_target
        self.black_point_target = black_point_target
        self.white_point_target = white_point_target
        self.output_power = output_power
        self.latitude = latitude
        self.contrast = contrast
        self.saturation = saturation
        self.balance = balance
        self.noise_level = noise_level
        self.sigma_toe = sigma_toe
        self.sigma_shoulder = sigma_shoulder
        self.preserve_color = preserve_color
        self.version = version
        self.auto_hardness = auto_hardness
        self.custom_grey = custom_grey
        self.high_quality_reconstruction = high_quality_reconstruction
        self.noise_distribution = noise_distribution
        self.shadows = shadows
        self.highlights = highlights
        self.compensate_icc_black = compensate_icc_black
        self.spline_version = spline_version
        self.enable_highlight_reconstruction = enable_highlight_reconstruction


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return np.clip(x, lo, hi)


def filmic_desaturate_v2(x: float, sigma_toe: float, sigma_shoulder: float, saturation: float) -> float:
    radius_toe = x
    radius_shoulder = 1.0 - x
    sat2 = 0.5 / np.sqrt(max(saturation, 0.0))
    key_toe = np.exp(-radius_toe * radius_toe / sigma_toe * sat2)
    key_shoulder = np.exp(-radius_shoulder * radius_shoulder / sigma_shoulder * sat2)
    return saturation - (key_toe + key_shoulder) * saturation


def linear_saturation(x: float, luminance: float, saturation: float) -> float:
    return luminance + saturation * (x - luminance)


def log_tonemapping_v2_1ch(x: float, grey: float, black: float, dynamic_range: float) -> float:
    return _clip((np.log2(x / grey) - black) / dynamic_range)


def exp_tonemapping_v2(x: float, grey: float, black: float, dynamic_range: float) -> float:
    return grey * np.exp2(dynamic_range * x + black)


def filmic_spline(
    x: float,
    M1: np.ndarray,
    M2: np.ndarray,
    M3: np.ndarray,
    M4: np.ndarray,
    M5: np.ndarray,
    latitude_min: float,
    latitude_max: float,
    type: np.ndarray,
) -> float:
    """Evaluate the filmic S-curve exactly as in the upstream snapshot."""
    M1 = np.asarray(M1, dtype=np.float32)
    M2 = np.asarray(M2, dtype=np.float32)
    M3 = np.asarray(M3, dtype=np.float32)
    M4 = np.asarray(M4, dtype=np.float32)
    M5 = np.asarray(M5, dtype=np.float32)
    type = np.asarray(type, dtype=np.int32)

    x = np.asarray(x, dtype=np.float32).reshape(-1)[0]
    if x < latitude_min:
        if type[0] == filmic_curve_type.poly_4:
            return M1[0] + x * (M2[0] + x * (M3[0] + x * (M4[0] + x * M5[0])))
        if type[0] == filmic_curve_type.poly_3:
            return M1[0] + x * (M2[0] + x * (M3[0] + x * M4[0]))
        # rational toe
        xi = latitude_min - x
        rat = xi * (xi * M2[0] + 1.0) + 0.0
        return M4[0] - M1[0] * rat / (rat + M3[0])
    if x > latitude_max:
        if type[1] == filmic_curve_type.poly_4:
            return M1[1] + x * (M2[1] + x * (M3[1] + x * (M4[1] + x * M5[1])))
        if type[1] == filmic_curve_type.poly_3:
            return M1[1] + x * (M2[1] + x * (M3[1] + x * M4[1]))
        # rational shoulder
        xi = x - latitude_max
        rat = xi * (xi * M2[1] + 1.0) + 0.0
        return M4[1] + M1[1] * rat / (rat + M3[1])
    return M1[2] + x * M2[2]


class FilmicRGBData:
    """In-memory state of the original filmic RGB module."""
    def __init__(self, params: filmicrgb_params) -> None:
        self.params = params
        self.spline: dict = {}


def _filmic_rgb_compute_spline(p: filmicrgb_params) -> dict:
    grey_display = pow(max(0.1845, p.grey_point_target / 100.0), 1.0 / p.output_power) if p.custom_grey else pow(0.1845, 1.0 / p.output_power)

    white_source = float(p.white_point_source)
    black_source = float(p.black_point_source)
    dynamic_range = white_source - black_source
    black_log = 0.0
    grey_log = abs(black_source) / dynamic_range if dynamic_range != 0.0 else 0.0
    white_log = 1.0

    if p.spline_version == filmic_spline_version.v1:
        black_display = _clip(p.black_point_target / 100.0)
        white_display = max(p.white_point_target, p.grey_point_target) / 100.0
    else:
        black_display = pow(_clip(p.black_point_target / 100.0), 1.0 / p.output_power)
        white_display = pow(max(p.white_point_target, p.grey_point_target) / 100.0, 1.0 / p.output_power)

    balance = _clip(p.balance / 100.0)
    toe_display = 0.0
    shoulder_display = 0.0
    if p.spline_version < filmic_spline_version.v3:
        latitude = _clip(p.latitude / 100.0 * dynamic_range)
        contrast = _clip(max(1.00001, p.contrast), 1.00001, 6.0)
        linear_intercept = grey_display - contrast * grey_log
        toe_log = grey_log - latitude / dynamic_range * abs(black_source / dynamic_range)
        shoulder_log = grey_log + latitude / dynamic_range * abs(white_source / dynamic_range)
        toe_display = toe_log * contrast + linear_intercept
        shoulder_display = shoulder_log * contrast + linear_intercept
        norm = np.sqrt(contrast * contrast + 1.0)
        coeff = -((2.0 * latitude) / dynamic_range) * balance
        toe_display += coeff * contrast / norm
        shoulder_display += coeff * contrast / norm
        toe_log += coeff / norm
        shoulder_log += coeff / norm
    else:
        hardness = p.output_power
        latitude = _clip(p.latitude / 100.0)
        slope = p.contrast * dynamic_range / 8.0
        min_contrast = 1.0
        min_contrast = max(min_contrast, (white_display - grey_display) / (white_log - grey_log))
        min_contrast = max(min_contrast, (grey_display - black_display) / (grey_log - black_log))
        min_contrast += 0.01
        contrast = slope / (hardness * pow(grey_display, hardness - 1.0))
        contrast = _clip(contrast, min_contrast, 100.0)
        linear_intercept = grey_display - contrast * grey_log
        xmin = (black_display + 0.01 * (white_display - black_display) - linear_intercept) / contrast
        xmax = (white_display - 0.01 * (white_display - black_display) - linear_intercept) / contrast
        toe_log = (1.0 - latitude) * grey_log + latitude * xmin
        shoulder_log = (1.0 - latitude) * grey_log + latitude * xmax
        balance_correction = 2.0 * balance * (shoulder_log - grey_log) if balance > 0.0 else 2.0 * balance * (grey_log - toe_log)
        toe_log -= balance_correction
        shoulder_log -= balance_correction
        toe_log = max(toe_log, xmin)
        shoulder_log = min(shoulder_log, xmax)

    return {
        "x": np.array([black_log, toe_log, grey_log, shoulder_log, white_log], dtype=np.float32),
        "y": np.array([black_display, toe_display, grey_display, shoulder_display, white_display], dtype=np.float32),
        "latitude_min": toe_log,
        "latitude_max": shoulder_log,
        "type": np.array([p.shadows, p.highlights], dtype=np.int32),
        "M1": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "M2": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "M3": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "M4": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "M5": np.array([0.0, 0.0, 0.0], dtype=np.float32),
    }


def _filmic_rgb_decompose(p: filmicrgb_params, spline: dict) -> tuple:
    """Return (lambda_min, lambda_max, toe_poly4, shoulder_poly4) from the upstream
    spline nodes and curve types."""
    M1 = spline["M1"]
    M2 = spline["M2"]
    M3 = spline["M3"]
    M4 = spline["M4"]
    M5 = spline["M5"]
    lam_min = float(spline["latitude_min"])
    lam_max = float(spline["latitude_max"])
    type_arr = spline["type"]

    if type_arr[0] == filmic_curve_type.poly_4:
        toe_poly4 = [float(M1[0]), float(M2[0]), float(M3[0]), float(M4[0]), float(M5[0])]
    else:
        toe_poly4 = [float(x) for x in M1]
    if type_arr[1] == filmic_curve_type.poly_4:
        shoulder_poly4 = [float(M1[1]), float(M2[1]), float(M3[1]), float(M4[1]), float(M5[1])]
    else:
        shoulder_poly4 = [float(x) for x in M1]

    return lam_min, lam_max, toe_poly4, shoulder_poly4


def filmic_rgb_unfiltered(
    pix_in: np.ndarray,
    data: FilmicRGBData,
    spline: dict,
    width: int,
    height: int,
) -> np.ndarray:
    """Unfiltered filmic RGB tonemap (no reconstruction)."""
    params = data.params
    width = int(width)
    height = int(height)

    grey_source = float(params.grey_point_source)
    black_source = float(params.black_point_source)
    dynamic_range = float(params.white_point_source - params.black_point_source)

    # Log tone mapping of the norm
    norm = np.maximum(pix_in, 0.0)
    norm = log_tonemapping_v2_1ch(norm, grey_source, black_source, dynamic_range)

    # Desaturation
    desaturation = filmic_desaturate_v2(norm, params.sigma_toe, params.sigma_shoulder, params.saturation)

    # Per-channel: apply spline to the desaturated, linear-saturated channel
    out = np.zeros_like(pix_in, dtype=np.float32)
    lam_min = float(spline["latitude_min"])
    lam_max = float(spline["latitude_max"])
    type_arr = spline["type"]
    M1 = spline["M1"]
    M2 = spline["M2"]
    M3 = spline["M3"]
    M4 = spline["M4"]
    M5 = spline["M5"]

    for c in range(3):
        value = norm[..., c] if norm.ndim == 3 else norm[c]
        # linear saturation on the original channel
        value = linear_saturation(pix_in[..., c], value, desaturation)
        value = filmic_spline(value, M1, M2, M3, M4, M5, lam_min, lam_max, type_arr)
        value = np.clip(value, 0.0, 1.0)
        out[..., c] = np.power(value, params.output_power)

    return out


def filmic_rgb(data: FilmicRGBData, pix_in: np.ndarray, params: filmicrgb_params | None = None) -> np.ndarray:
    """Apply the original filmic RGB tone map."""
    if params is not None:
        data.params = params
    if not data.spline:
        data.spline = _filmic_rgb_compute_spline(data.params)
    return filmic_rgb_unfiltered(pix_in, data, data.spline, 1, 1)
