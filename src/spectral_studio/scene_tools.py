"""Scene-linear exposure utilities; NOT a port of darktable algorithms.
References: https://github.com/darktable-org/darktable/blob/master/src/iop/exposure.c
"""
import numpy as np

def exposure(rgb, ev):
    a=np.asarray(rgb,dtype=np.float32)
    if not np.isfinite(a).all(): raise ValueError('nonfinite image')
    return a * np.float32(2.0 ** float(ev))

def suggest_exposure(rgb, percentile=50., target_ev=-4.):
    """Generic image percentile estimator, not darktable RAW deflicker."""
    a=np.asarray(rgb,dtype=np.float32)
    if a.ndim!=3 or a.shape[-1]!=3: raise ValueError('RGB expected')
    luminance=.2126*a[:,:,0]+.7152*a[:,:,1]+.0722*a[:,:,2]
    valid=luminance[np.isfinite(luminance)&(luminance>0)]
    if not valid.size: raise ValueError('no valid positive luminance')
    sample=float(np.percentile(valid,percentile))
    return float(np.clip(target_ev-np.log2(sample),-8,8))
