"""Derive the 2,400-row discovery variant (…x4d1) from saved …x4d4 predictions: same model, first 2,400 discovery rows.
Usage: python make_x4_small.py <folder with *x4d4* files>"""
import sys, glob, os, numpy as np
for f in glob.glob(os.path.join(sys.argv[1], "*x4d4_seed*_ctxfull.npz")):
    z = dict(np.load(f)); z["zd"] = z["zd"][:2400]; z["yd"] = z["yd"][:2400]
    z["rows_predicted"] = np.int64(2400 + len(z["zs"]) + len(z["zt"]))
    np.savez_compressed(f.replace("x4d4", "x4d1"), **z)
print("ok")
