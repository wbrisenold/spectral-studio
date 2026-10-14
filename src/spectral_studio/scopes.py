"""Image measurements; values refer to the buffer passed by the caller."""
import numpy as np
def rgb_array(image):
    a=np.asarray(image,dtype=np.float32)
    if a.ndim!=3 or a.shape[-1]!=3: raise ValueError('H x W x 3 required')
    return np.nan_to_num(a,nan=0,posinf=0,neginf=0)
def histogram(image,bins=256):
    a=rgb_array(image)
    return np.stack([np.histogram(a[:,:,c],bins=bins,range=(0,1))[0] for c in range(3)])
def waveform(image,bins=256):
    a=rgb_array(image); h,w,_=a.shape
    x=np.broadcast_to((np.arange(w)*bins//w)[None,:],(h,w))
    y=np.clip((1-(.2126*a[:,:,0]+.7152*a[:,:,1]+.0722*a[:,:,2]))*(bins-1),0,bins-1).astype(int)
    out=np.zeros((bins,bins),dtype=np.int64)
    np.add.at(out,(y,x),1)
    return out
def parade(image,bins=256):
    a=rgb_array(image); h,w,_=a.shape
    x=np.broadcast_to((np.arange(w)*bins//w)[None,:],(h,w))
    out=np.zeros((3,bins,bins),dtype=np.int64)
    for c in range(3):
        y=np.clip((1-a[:,:,c])*(bins-1),0,bins-1).astype(int)
        np.add.at(out[c],(y,x),1)
    return out
def vectorscope(image,bins=256):
    a=rgb_array(image)
    y=.2126*a[:,:,0]+.7152*a[:,:,1]+.0722*a[:,:,2]
    cb=(a[:,:,2]-y)/1.8556; cr=(a[:,:,0]-y)/1.5748
    return np.histogram2d(cb.ravel(),cr.ravel(),bins=bins,range=[[-.5,.5],[-.5,.5]])[0]
def clipping(image):
    a=rgb_array(image)
    return float(np.mean(np.any(a>=1,axis=2))*100),float(np.mean(np.all(a<=0,axis=2))*100)
