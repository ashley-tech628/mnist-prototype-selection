"""Local IDX reading and a class-balance ablation; not the historical PCA/k-means run."""
from pathlib import Path
import struct
import numpy as np


def read_idx(path):
    raw=Path(path).read_bytes()
    if len(raw)<4 or raw[:3]!=b'\0\0\x08':raise ValueError('Expected uint8 IDX')
    dimensions=raw[3]
    if dimensions not in (1,3) or len(raw)<4+4*dimensions:raise ValueError('Unsupported IDX dimensions')
    shape=struct.unpack('>'+('I'*dimensions),raw[4:4+4*dimensions])
    offset=4+4*dimensions
    if len(raw)-offset!=int(np.prod(shape)):raise ValueError('IDX payload size mismatch')
    return np.frombuffer(raw,dtype=np.uint8,offset=offset).reshape(shape)


def balanced_random(y,budget,seed):
    if not 0<budget<=len(y):raise ValueError('Budget must be in [1,N]')
    rng=np.random.default_rng(seed)
    classes=np.unique(y)
    chosen=[]
    # Initial equal quotas; refill from remaining training rows if a class is short.
    for j,c in enumerate(classes):
        available=np.flatnonzero(y==c)
        count=min(len(available),budget//len(classes)+(j<budget%len(classes)))
        chosen.extend(rng.choice(available,count,replace=False).tolist())
    remaining=np.setdiff1d(np.arange(len(y)),chosen)
    if len(chosen)<budget:chosen.extend(rng.choice(remaining,budget-len(chosen),replace=False).tolist())
    return np.asarray(chosen,dtype=np.int64)
