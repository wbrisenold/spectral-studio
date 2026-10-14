"""RAW ingestion via rawpy/LibRaw, producing scene-linear camera RGB.

This module does not claim a camera-to-working-space transform. The returned
array is camera RGB; film simulation must not consume it until its input color
contract is established and the appropriate transform is implemented.
"""
from pathlib import Path
import numpy as np

RAW_EXTENSIONS = frozenset({'.dng','.cr2','.cr3','.nef','.nrw','.arw','.srf','.sr2','.raf','.orf','.rw2','.pef','.3fr','.fff','.iiq','.k25','.kdc','.mos','.mrw','.raw','.rwl','.x3f'})

def is_raw(path):
    return Path(path).suffix.lower() in RAW_EXTENSIONS

def decode_raw(path, *, half_size=False):
    """Decode a RAW without auto-brightening or an sRGB transfer curve.

    rawpy postprocess returns a 16-bit RGB image. Normalizing by 65535 is a
    numeric conversion, not proof of calibrated scene-linear radiometry.
    """
    import rawpy
    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(
            output_color=rawpy.ColorSpace.raw,
            output_bps=16,
            gamma=(1,1),
            no_auto_bright=True,
            use_camera_wb=True,
            half_size=half_size,
        )
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('RAW decoder did not return three channels')
    return np.asarray(rgb, dtype=np.float32) / np.float32(65535.0)
