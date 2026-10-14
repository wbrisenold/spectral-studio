import numpy as np
from spectral_studio.scene_tools import exposure,suggest_exposure

def test_exposure_preserves_highlights():
    x=np.array([[[2.,1.,.5]]],dtype=np.float32)
    assert np.allclose(exposure(x,1),[[[4.,2.,1.]]])

def test_suggestion():
    x=np.ones((8,8,3),dtype=np.float32)*.125
    assert np.isclose(suggest_exposure(x,50,-4),-1)
