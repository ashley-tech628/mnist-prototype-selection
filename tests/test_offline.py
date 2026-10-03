import tempfile
from pathlib import Path
import struct
import unittest
import numpy as np
from prototype_lab.offline import read_idx,balanced_random
from knn_eval import knn_1nn_accuracy


class OfflineTests(unittest.TestCase):
    def test_batching_matches_direct_nearest_neighbor(self):
        rng=np.random.default_rng(1);x=rng.normal(size=(12,5)).astype(np.float32);xt=rng.normal(size=(7,5)).astype(np.float32)
        y=np.arange(12)%3;truth=np.arange(7)%3
        predictions=y[np.argmin(((xt[:,None]-x[None])**2).sum(-1),axis=1)]
        for batch in [1,3,20]:self.assertEqual(knn_1nn_accuracy(x,y,xt,truth,batch),float((predictions==truth).mean()))
    def test_class_balance_unique_and_deterministic(self):
        y=np.repeat(np.arange(3),20);idx=balanced_random(y,11,42)
        self.assertEqual(len(set(idx)),11)
        self.assertEqual(sorted(np.bincount(y[idx]).tolist()),[3,4,4])
        np.testing.assert_array_equal(idx,balanced_random(y,11,42))
    def test_short_class_refill(self):
        y=np.array([0]+[1]*20);idx=balanced_random(y,10,1)
        self.assertEqual(len(set(idx)),10);self.assertIn(0,idx)
    def test_invalid_budget(self):
        for budget in [0,-1,5]:
            with self.assertRaises(ValueError):balanced_random(np.array([0,1]),budget,1)
    def test_idx_contract(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'labels';p.write_bytes(struct.pack('>II',2049,3)+bytes([1,2,3]))
            np.testing.assert_array_equal(read_idx(p),[1,2,3])
            p.write_bytes(struct.pack('>II',2049,4)+bytes([1,2,3]))
            with self.assertRaises(ValueError):read_idx(p)


if __name__=='__main__':unittest.main()
