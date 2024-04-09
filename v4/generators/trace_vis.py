import os, sys
sys.path.append(os.path.join(os.environ.get("nas"), "code/vessel.da/"))

from v3.trace_algorithms.astar import Astar
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy import ndimage


def norm_(arr):
    return arr / np.linalg.norm(arr)


def generate_circle(arr, O, r):
    gen = arr.copy() * 0
    for p in range(arr.shape[0]):
        for q in range(arr.shape[1]):
            gen[p, q] = (((np.array([p, q]) - O) / np.array(r)) ** 2).sum() <= 1
    return gen


def generate_cluster(arr, O, r):
    gen = arr.copy() * 0
    for p in range(arr.shape[0]):
        for q in range(arr.shape[1]):
            for s in range(arr.shape[2]):
                gen[p, q, s] = (((np.array([p, q, s]) - O) / np.array(r)) ** 2).sum() <= 1
    return gen
    
    
def _bspline(path, degree=3, _astar=None, mask=None):
    path = np.array(path)
    n_samples = len(path)
        
    sampled_knots = np.linspace(0, 1, n_samples)
    sampled_knots = np.array([path[round(_ * (len(path) - 1))] for _ in sampled_knots]).T
    tck, _ = splprep(sampled_knots, k=degree, s=len(sampled_knots[0]) + np.sqrt(sampled_knots.shape[1]))
    path = np.linspace(0, 1, len(path))
    path = np.array(splev(path, tck)).T  # .astype(int)
    
    """filled_path = []
    path = curve_points
    for ip in range(len(path) - 1):
        s_ = path[ip]
        e_ = path[ip + 1]
        if len(filled_path) > 0:
            filled_path.pop(-1)
        filled_path.extend(_astar.connect(s_, e_, e_ - s_))"""
    return path


fig = plt.figure(figsize=(17, 8))

# planar
canvas = np.zeros((100, 100))
start = np.zeros((2,))
end = np.array(canvas.shape) - 1
start_dir = norm_(np.array([1, 1]))

organ_mask = generate_circle(canvas, O=(60, 64), r=(15, 15)) + generate_circle(canvas, O=(10, 10), r=(4, 7))
organ_mask_ero = ndimage.binary_erosion(organ_mask > 0, iterations=2)
a_star = Astar(organ_mask_ero, p=0)
our_star = Astar(organ_mask, p=.4)
a_star_trace = np.array(a_star.connect(start, end, start_dir)) #, _astar=a_star, mask=)
our_star_tr_ = np.array(our_star.connect(start, end, start_dir))
our_star_trace = _bspline(our_star_tr_, _astar=our_star, mask=organ_mask)

ax = fig.add_subplot(1, 2, 1)
ax.scatter(*np.argwhere(organ_mask).T, s=1, c="#c0c0c0", label="organ buffer")
ax.scatter(*np.argwhere(ndimage.binary_erosion(organ_mask > 0, iterations=2)).T, s=1, c='k', label="organ mask")
# ax.plot(*a_star_trace.T, lw=1, c='b')
ax.plot(*a_star_trace.T, lw=1., c='g')
ax.plot(*our_star_trace.T, lw=2, c='r')
ax.set_xticklabels([])
ax.set_yticklabels([])

# volumetric
canvas = np.zeros((100, 100, 100))
start = np.zeros((3,))
end = np.array(canvas.shape) - 1
start_dir = norm_(np.array([1, 1, 1]))

organ_mask = generate_cluster(canvas, O=(50, 50, 50), r=(15,)) + generate_cluster(canvas, O=(25, 25, 10), r=(4, 7, 2))
organ_mask_ero = ndimage.binary_erosion(organ_mask > 0, iterations=2)
a_star = Astar(organ_mask_ero, p=0)
our_star = Astar(organ_mask, p=.4)
a_star_trace = np.array(a_star.connect(start, end, start_dir)) #, _astar=a_star, mask=)
our_star_tr_ = np.array(our_star.connect(start, end, start_dir))
our_star_trace = _bspline(our_star_tr_, _astar=our_star, mask=organ_mask)

ax = fig.add_subplot(1, 2, 2, projection="3d")
ax.scatter(*np.argwhere(organ_mask).T, s=1, c="#c0c0c0", alpha=0.1)
ax.scatter(*np.argwhere(ndimage.binary_erosion(organ_mask > 0, iterations=2)).T, s=1, c='k', label="organ mask", alpha=0.1)
# ax.plot(*a_star_trace.T, lw=1, c='b')
ax.plot(*a_star_trace.T, lw=1., c='g')
ax.plot(*our_star_trace.T, lw=2, c='r')
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.xaxis.set_pane_color((0.2, 0.2, 0.2, 0.0))
ax.yaxis.set_pane_color((0.2, 0.2, 0.2, 0.0))
ax.zaxis.set_pane_color((0.2, 0.2, 0.2, 0.0))

plt.draw()
plt.savefig("v3/imgs/trace.png", dpi=300, bbox_inches="tight")