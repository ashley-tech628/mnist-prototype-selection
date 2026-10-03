"""Compare uniform and class-balanced random selection on a local MNIST subset."""
from pathlib import Path
import argparse
import json
import hashlib
import platform
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prototype_lab.offline import read_idx,balanced_random
from knn_eval import knn_1nn_accuracy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,required=True,help='MNIST/raw directory with uncompressed IDX files')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--budget',type=int,default=300)
    p.add_argument('--train-size',type=int,default=5000)
    p.add_argument('--test-size',type=int,default=1000)
    a=p.parse_args()
    if a.out.exists() and any(a.out.iterdir()):p.error('Output directory must be empty')
    if min(a.train_size,a.test_size,a.budget)<1:p.error('Sizes must be positive')
    names=['train-images-idx3-ubyte','train-labels-idx1-ubyte','t10k-images-idx3-ubyte','t10k-labels-idx1-ubyte']
    images,labels,test_images,test_labels=[read_idx(a.raw/n) for n in names]
    if len(images)!=len(labels) or len(test_images)!=len(test_labels):p.error('Image/label size mismatch')
    rng=np.random.default_rng(a.seed)
    ti=rng.choice(len(images),min(a.train_size,len(images)),replace=False)
    ei=rng.choice(len(test_images),min(a.test_size,len(test_images)),replace=False)
    x=images[ti].reshape(len(ti),-1).astype(np.float32)/255
    y=labels[ti];xt=test_images[ei].reshape(len(ei),-1).astype(np.float32)/255;yt=test_labels[ei]
    if a.budget>len(x):p.error('Budget exceeds selected training set')
    report={'kind':'new subset class-balance ablation; not historical PCA/kmeans reproduction',
            'train_size':len(x),'test_size':len(xt),'budget':a.budget,'seed':a.seed,
            'features':'784 normalized raw pixels; no PCA',
            'train_indices_sha256':hashlib.sha256(ti.astype('<i8').tobytes()).hexdigest(),
            'test_indices_sha256':hashlib.sha256(ei.astype('<i8').tobytes()).hexdigest(),
            'dataset_sha256':{n:hashlib.sha256((a.raw/n).read_bytes()).hexdigest() for n in names},
            'environment':{'python':platform.python_version(),'numpy':np.__version__},'methods':{}}
    for method in ['uniform_random','class_balanced_random']:
        start=time.perf_counter()
        idx=np.random.default_rng(a.seed).choice(len(x),a.budget,replace=False) if method=='uniform_random' else balanced_random(y,a.budget,a.seed)
        selection_seconds=time.perf_counter()-start
        start=time.perf_counter();accuracy=knn_1nn_accuracy(x[idx],y[idx],xt,yt,batch=128)
        report['methods'][method]={'accuracy':accuracy,'selection_seconds':selection_seconds,
            'prediction_seconds':time.perf_counter()-start,'feature_bytes':int(x[idx].nbytes),
            'distance_matrix_upper_bound_bytes':128*a.budget*4,
            'class_counts':{str(c):int((y[idx]==c).sum()) for c in np.unique(y)}}
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
