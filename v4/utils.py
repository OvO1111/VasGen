import os
import time
import shutil
import numpy as np
import scipy.ndimage as ndimage
import SimpleITK as sitk

from scipy import ndimage
from functools import reduce
from os.path import exists, join
from collections import namedtuple
from collections.abc import Iterable


def listdir(path, ignore_patterns='.', select_func=None):
    if select_func is None: select_func = lambda _: True
    return sorted([join(path, subpath) for subpath in os.listdir(path) if not subpath.startswith(ignore_patterns) and select_func(subpath)])


def makedir_or_dirs(path, destory_on_exist=False):
    if exists(path):
        if destory_on_exist:
            shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    return path


def bounding_box(array, pt=None, outline=0):
    nd = len(array.shape)
    if nd == 2:
        H, W = array.shape
        if isinstance(outline, (int, float)):
            outline = [int(outline)] * 4
        if isinstance(outline, list) and len(outline) == 2:
            outline = reduce(lambda a, b: a + b, [outline] * 2)
        
        try:
            if pt is None:
                h = np.any(array, axis=1)
                w = np.any(array, axis=0)

                hmin, hmax = np.where(h)[0][[0, -1]]
                wmin, wmax = np.where(w)[0][[0, -1]]
            else:
                hmin = hmax = pt[0]
                wmin = wmax = pt[1]
            
            bbox = [[max(hmin-outline[0], 0), min(hmax+outline[1], H)+1],
                    [max(wmin-outline[2], 0), min(wmax+outline[3], W)+1],]
        except Exception as e:
            bbox = None
            print(e)
            
    if nd == 3:
        H, W, D = array.shape
        if isinstance(outline, (int, float)):
            outline = [int(outline)] * 6
        if isinstance(outline, Iterable) and len(outline) == 3:
            outline = np.array(list(zip(outline, outline))).flatten().tolist()
        
        try:
            if pt is None:
                h = np.any(array, axis=(1, 2))
                w = np.any(array, axis=(0, 2))
                d = np.any(array, axis=(0, 1))

                hmin, hmax = np.where(h)[0][[0, -1]]
                wmin, wmax = np.where(w)[0][[0, -1]]
                
                dmin, dmax = np.where(d)[0][[0, -1]]
            else:
                hmin = hmax = pt[0]
                wmin = wmax = pt[1]
                dmin = dmax = pt[2]
            
            bbox = [[max(hmin-outline[0], 0), min(hmax+outline[1], H)],
                    [max(wmin-outline[2], 0), min(wmax+outline[3], W)],
                    [max(dmin-outline[4], 0), min(dmax+outline[5], D)]]
        except Exception as e:
            bbox = None
            print(e)
        
    def transformer(point_or_array):
        point_or_array = np.round(point_or_array).astype(int)
        if point_or_array.ndim == nd:
            return bounding_box(point_or_array, pt, outline)
        else:
            if point_or_array.ndim == 1:
                point_or_array = point_or_array[None]
            if point_or_array.shape[1] != nd:
                point_or_array = point_or_array.T
            point_or_array -= np.array([[bbox[_][0] for _ in range(len(bbox))]])
            if len(point_or_array) == 1:
                return point_or_array[0]
            return point_or_array
    
    def inv_transformer(point_or_array, fill=0):
        point_or_array = np.round(point_or_array).astype(int)
        if point_or_array.ndim == nd:
            if nd == 3:
                return np.pad(point_or_array, ((hmin, H-hmax), (wmin, W-wmax), (dmin, D-dmax)), constant_values=fill)
            if nd == 2:
                return np.pad(point_or_array, ((hmin, H-hmax), (wmin, W-wmax)), constant_values=fill)
        else:
            if point_or_array.ndim == 1:
                point_or_array = point_or_array[None]
            if point_or_array.shape[1] != nd:
                point_or_array = point_or_array.T
            point_or_array += np.array([[bbox[_][0] for _ in range(len(bbox))]])
            if len(point_or_array) == 1:
                return point_or_array[0]
            return point_or_array
        
    def cropper():
        if bbox is None: return [slice(0, 1, None)] * nd
        else: return list(map(lambda x: slice(x[0], x[1], None), bbox))
        
    BoundingBox = namedtuple("BoundingBox", ["bbox", "cropper", "transformer", "inv_transformer"])
    return BoundingBox(bbox, 
                       cropper(),
                       lambda arr: transformer(arr),
                       lambda arr, fill=0: inv_transformer(arr, fill))
    
    
def window_norm(array, window_pos=0, window_len=0, typename=None):
    if typename == "abdomen_vessel":
        window_pos = 60
        window_len = 360
        
    window_lower_bound = window_pos - window_len // 2
    array = (array - window_lower_bound) / window_len
    array[array < 0] = 0
    array[array > 1] = 1
    return array


def soft_skeleton3d(inputs: np.ndarray, itr: int):
    s = inputs & ~ndimage.binary_opening(inputs)
    for _ in range(itr):
        inputs = ndimage.binary_erosion(inputs)
        inputs_ = ndimage.binary_opening(inputs)
        s_ = inputs & ~inputs_
        s = s | (s_ & ~(s & s_))
    return s


def timeit(func):
    def _impl(*args, **kwargs):
        start = time.time()
        retval = func(*args, **kwargs)
        elapsed = time.time() - start
        return retval, elapsed
    return _impl


def find_largest_connected_components(mask, p=None, n=None, i0=0):
    if mask.sum() == 0:
        return mask
    labeled_mask, num_labels = ndimage.label(mask)
    
    num_label_pixels = np.bincount(labeled_mask.flatten())
    num_label_pixels = [(l, num_label_pixels[l]) for l in range(1, num_labels+1)]
    label_nums = np.array(num_label_pixels)
    if p is not None:
        threshold = p * label_nums[:, 1].max()
        lcc = [_[0] for _ in num_label_pixels if _[1] >= threshold]
    elif n is not None:
        num_label_pixels.sort(reverse=True, key=lambda x: x[1])
        lcc = [num_label_pixels[i][0] for i in range(n)]
    else:
        lcc = [num_label_pixels[label_nums[:, 1].argmax()][0]]
    
    res = np.zeros(mask.shape, dtype=np.uint8)
    res_vein_art_label = [(labeled_index, np.argwhere(labeled_mask == labeled_index).mean(axis=0)) for labeled_index in lcc]
    if len(mask.shape) == 3:
        res_vein_art_label.sort(key=lambda x: x[1][2], reverse=True)  # sort from L to R
    for i, (labeled_index, _) in enumerate(res_vein_art_label):
        res[labeled_mask == labeled_index] = i0 + i + 1
    return res


def get_direction(path, por_or_index=-1):
    
    if len(path) <= 1:
        raise RuntimeError("length of the given path should be more than 1")
    
    path = np.array(path)
    if isinstance(por_or_index, float): cutoff = round(len(path) * por_or_index)
    elif isinstance(por_or_index, int): cutoff = por_or_index if por_or_index > 0 else len(path) + por_or_index
    path_ext = np.pad(path, ((1, 1), (0, 0)), "edge")
    decay_fn = lambda x: 1 / (x / 2 + 1)
    
    former_part = path[:cutoff + 1]
    latter_part = path[cutoff:]
    
    s = former_part - path_ext[:cutoff + 1]
    t = path_ext[cutoff + 2:] - latter_part
    c = (s * decay_fn(np.arange(0, cutoff+1)[::-1, None]) + t * decay_fn(np.arange(0, len(path)-cutoff))[None]).sum(0)
    direction = c / np.linalg.norm(c)
    
    """s = np.asarray([-(path[:cutoff] - path[cutoff]) * np.exp(-np.linspace(0, cutoff, cutoff))[::-1, np.newaxis]]).sum(axis=0)
    t = np.asarray([(path[cutoff+1:] - path[cutoff]) * np.exp(-np.linspace(0, len(path)-cutoff-1, len(path)-cutoff-1))[:, np.newaxis]]).sum(axis=0)
    direction = np.vstack([s, t]).mean(axis=0)
    direction /= np.linalg.norm(direction)"""
    return direction


def oriented_segment(array, code, percentile=50, orient=-1, direction=-1, half=-1):
    if code is not None:
        if code == 'L': orient, direction, half = (1, 2, 0)
        elif code == 'R': orient, direction, half = (1, 2, 1)
        elif code == "A": orient, direction, half = (2, 1, 0)
        elif code == 'P': orient, direction, half = (2, 1, 1)
        elif code == 'S': orient, direction, half = (2, 0, 0)
        elif code == 'I': orient, direction, half = (2, 0, 1)
    else: 
        if orient not in [0, 1, 2]: return array
    
    array_copy = array.copy()
    axes = [0, 1, 2]
    axes.remove(orient)
    axis = np.argwhere(axes == orient)
    
    for layer in range(array_copy.shape[orient]):
        if orient == 0:
            array_slice = array_copy[layer]
        elif orient == 1:
            array_slice = array_copy[:, layer]
        else:
            array_slice = array_copy[:, :, layer]
        axis = 0 if direction == 0 else 1
        
        mean = np.argwhere(np.any(array_slice > 0, axis=1-axis))
        if len(mean) == 0: mean = 0 if half == 0 else -1
        else: mean = np.percentile(mean, percentile)
        retrieve = slice(0, round(mean)) if half == 0 else slice(round(mean), None)
        if axis == 0: array_slice[retrieve] = 0
        else: array_slice[:, retrieve] = 0
    
    array[array_copy > 0] = False
    return array


class NiftiReader:
    def __init__(self, nifti_path=None):
        if nifti_path is not None: self.load(nifti_path)
        
    def load(self, nifti_path, save_path=None):
        self.raw_nifti_path = nifti_path
        self.save_array_path = save_path
        self.raw_nifti_image = sitk.ReadImage(self.raw_nifti_path)
        return self.array
        
    @property
    def array_view(self) -> np.ndarray:
        return sitk.GetArrayViewFromImage(self.raw_nifti_image).transpose(2, 1, 0)
    
    @property
    def array(self) -> np.ndarray:
        return sitk.GetArrayFromImage(self.raw_nifti_image).transpose(2, 1, 0)
    
    @property
    def spacing(self) -> np.ndarray:
        return np.array(self.raw_nifti_image.GetSpacing())
    
    @property
    def zeros(self) -> np.ndarray: 
        return lambda **kwargs: np.zeros(self.array_view.shape, **kwargs)
    
    def save_to_disk(self, array, path=None):
        if path is None: path = self.save_array_path
        processed_nifti_image = sitk.GetImageFromArray(array.transpose(2, 1, 0))
        assert processed_nifti_image.GetDimension() == self.raw_nifti_image.GetDimension()
        
        processed_nifti_image.SetOrigin(self.raw_nifti_image.GetOrigin())
        processed_nifti_image.SetSpacing(self.raw_nifti_image.GetSpacing())
        processed_nifti_image.SetDirection(self.raw_nifti_image.GetDirection())
        sitk.WriteImage(processed_nifti_image, path)
        print(f"wrote image to {path}")
        
        
def cldice(v_p, v_l):
    from skimage.morphology import skeletonize, skeletonize_3d
    import numpy as np

    def cl_score(v, s):
        return np.sum(v*s) / np.sum(s)

    if len(v_p.shape) == 2:
        tprec = cl_score(v_p, skeletonize(v_l))
        tsens = cl_score(v_l, skeletonize(v_p))
    elif len(v_p.shape) == 3:
        tprec = cl_score(v_p, skeletonize_3d(v_l))
        tsens = cl_score(v_l, skeletonize_3d(v_p))
    if (tprec + tsens) == 0: return 0
    return 2*tprec*tsens/(tprec+tsens)


def simple_visualize(arr: np.ndarray, path="./simple_visualization.nii.gz"):
    if arr.dtype == bool: arr = arr.astype(np.uint8)
    sitk.WriteImage(sitk.GetImageFromArray(arr), path)


if __name__ == "__main__":
    line = np.array([[0, 0, _] for _ in range(10)] + [[0, 1, 9]])
    d = get_direction(line, -1)